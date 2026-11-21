"""SVDQ-FP8 (W8A8) quantization: packed runtime module and offline quantizer.

The format mirrors SVDQuant's W4A4 layout but keeps the residual in E4M3 with a single
per-output-channel FP32 scale, so no group scales or packed nibble layouts are involved::

    W * s = proj_up @ (proj_down * s) + residual
    qweight[n,k] = satfinite_e4m3(residual[n,k] / wscales[n])
    wscales[n]   = amax_k |residual[n,:]| / 448
    proj_down    = L2 / s

The forward pass is fully fused: `svdq_quantize_w8a8` produces the per-token E4M3 activation and the
BF16 LoRA-down term in one kernel, and `svdq_gemm_w8a8` adds the E4M3 product and the LoRA-up term
into the same FP32 accumulator. There is no standalone BF16 GEMM and no second pass over the dense
output.
"""

import torch
from torch import nn

from ...kernels import svdq_gemm_w8a8
from ...kernels import svdq_quantize_w8a8
from .lowrank import decompose_lowrank_residual
from .quantizer import _compute_weight_only_smooth_scale
from .quantizer import _normalize_dtype
from .quantizer import compute_smooth_scale

FP8_EPS = 1e-12
SVDQ_FP8_RANKS = (0, 32, 64, 128)


def _pad_to(value: int, multiple: int) -> int:
  return (value + multiple - 1) // multiple * multiple


def quantize_residual_fp8_per_row(residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Quantize a residual weight matrix to E4M3 with one FP32 scale per output channel.

  :param residual: Floating-point residual ``[N,K]`` after the low-rank split.
  :returns: ``(qweight[N,K] e4m3, wscales[N] fp32)`` where ``wscales`` is the dequantization
    multiplier ``amax/448`` clamped away from zero so empty rows stay finite.
  """
  residual = residual.float()
  scale = residual.abs().amax(dim=1, keepdim=True).clamp_min(FP8_EPS) / 448.0
  qweight = (residual / scale).to(torch.float8_e4m3fn)
  return qweight, scale.squeeze(1).contiguous()


class SVDQW8A8Linear(nn.Module):
  """Runtime module for packed SVDQ-FP8 linear layers.

  Buffers are stored padded to the kernel's 128-wide geometry; the original feature sizes are kept
  so `forward` can slice the result back. `proj_down` already contains the inverse smoothing factor,
  which is what allows the fused producer to run the low-rank branch on the raw activation.
  """

  def __init__(
    self,
    in_features: int,
    out_features: int,
    rank: int = 32,
    bias: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
  ) -> None:
    """Initialize an empty packed W8A8 module.

    :param in_features: Logical input width of the source linear.
    :param out_features: Logical output width of the source linear.
    :param rank: Low-rank correction rank; one of 0, 32, 64 or 128.
    :param bias: Whether to keep a BF16 bias.
    :param torch_dtype: Dtype of the bias and low-rank factors. Only bfloat16 is implemented.
    :param device: Allocation device; CPU when omitted.
    """
    super().__init__()
    if rank not in SVDQ_FP8_RANKS:
      raise ValueError(f"SVDQ-FP8 supports rank 0/32/64/128, got {rank}.")
    if torch_dtype != torch.bfloat16:
      raise ValueError("SVDQW8A8Linear currently requires torch_dtype=torch.bfloat16.")
    if device is None:
      device = torch.device("cpu")
    self.in_features = in_features
    self.out_features = out_features
    self.rank = rank
    self.quantized_in_features = _pad_to(in_features, 128)
    self.quantized_out_features = _pad_to(out_features, 128)
    kp, np_ = self.quantized_in_features, self.quantized_out_features
    self.register_buffer("qweight", torch.zeros((np_, kp), dtype=torch.float8_e4m3fn,
                                                device=device))
    self.register_buffer("wscales", torch.zeros(np_, dtype=torch.float32, device=device))
    self.register_buffer("smooth_factor", torch.ones(kp, dtype=torch_dtype, device=device))
    self.register_buffer("proj_down", torch.zeros((rank, kp), dtype=torch_dtype, device=device))
    self.register_buffer("proj_up", torch.zeros((np_, rank), dtype=torch_dtype, device=device))
    self.bias = nn.Parameter(torch.zeros(np_, dtype=torch_dtype, device=device),
                             requires_grad=False) if bias else None

  @classmethod
  def from_linear(
    cls,
    linear: nn.Linear,
    *,
    rank: int = 32,
    smooth_factor: torch.Tensor | None = None,
    alpha: float = 0.5,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
    calibrate_precision: str = "low",
  ) -> "SVDQW8A8Linear":
    """Quantize a float linear layer into the packed W8A8 format.

    :param linear: Source floating-point linear layer.
    :param rank: Low-rank correction rank.
    :param smooth_factor: Optional calibration-derived per-input-channel smoothing vector. When
      omitted, a weight-only SmoothQuant factor with exponent `alpha` is used (the DQ path).
    :param alpha: SmoothQuant interpolation factor for the weight-only fallback.
    :param torch_dtype: Dtype for the bias and low-rank factors.
    :param device: Device on which to materialize the packed module.
    :param calibrate_precision: Precision policy for the low-rank SVD (`low`/`medium`/`high`).
    :returns: The packed module.
    """
    device = torch.device(device or linear.weight.device)
    module = cls(linear.in_features,
                 linear.out_features,
                 rank=rank,
                 bias=linear.bias is not None,
                 torch_dtype=torch_dtype,
                 device=device)
    weight = linear.weight.detach().to(device=device, dtype=torch.float32)
    n, k = weight.shape
    if smooth_factor is None:
      weight_span = weight.abs().amax(dim=0)
      smooth = _compute_weight_only_smooth_scale(weight_span,
                                                 alpha=alpha,
                                                 math_dtype=torch.float32,
                                                 output_dtype=torch.float32)
    else:
      smooth = smooth_factor.to(device=device, dtype=torch.float32)
    if smooth.numel() != k:
      raise ValueError(f"smooth_factor must have {k} entries, got {smooth.numel()}.")
    smoothed = weight * smooth.unsqueeze(0)
    down, up, residual = decompose_lowrank_residual(smoothed,
                                                    rank,
                                                    output_dtype=torch.float32,
                                                    svd_precision=calibrate_precision)
    qweight, wscales = quantize_residual_fp8_per_row(residual)
    # The inverse smoothing is folded in float64 exactly like the W4A4 exporter: dropping this
    # division would leave the whole low-rank branch off by one factor of s.
    proj_down = (down.to(torch.float64) / smooth.to(torch.float64).unsqueeze(0)).to(torch_dtype)

    module.qweight.zero_()
    module.wscales.zero_()
    module.proj_up.zero_()
    module.proj_down.zero_()
    module.smooth_factor.fill_(1.0)
    module.qweight[:n, :k] = qweight
    module.wscales[:n] = wscales
    module.smooth_factor[:k] = smooth.to(torch_dtype)
    if rank:
      module.proj_down[:, :k] = proj_down
      module.proj_up[:n, :] = up.to(torch_dtype)
    if module.bias is not None:
      module.bias.zero_()
      module.bias[:n] = linear.bias.detach().to(device=device, dtype=torch_dtype)
    return module

  def dequantize_weight(self) -> torch.Tensor:
    """Return the effective full-precision weight, sliced back to the logical shape.

    :returns: Diagnostic reconstruction ``[out_features, in_features]``; not used by `forward`.
    """
    n, k = self.out_features, self.in_features
    residual = self.qweight[:n, :k].float() * self.wscales[:n, None]
    smooth = self.smooth_factor[:k].float()
    weight = residual / smooth.unsqueeze(0)
    if self.rank:
      weight = weight + self.proj_up[:n].float() @ self.proj_down[:, :k].float()
    return weight

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Run the fused producer + GEMM chain.

    :param x: Activation whose last dimension equals `in_features`.
    :returns: BF16 output with `out_features` channels and the same leading dimensions as `x`.
    """
    shape = x.shape
    rows = x.numel() // shape[-1]
    if rows == 0:
      return x.new_empty((*shape[:-1], self.out_features))
    flat = x.reshape(rows, shape[-1])
    if flat.shape[1] != self.quantized_in_features:
      flat = torch.nn.functional.pad(flat, (0, self.quantized_in_features - flat.shape[1]))
    flat = flat.contiguous()
    qact, ascales, lora_act = svdq_quantize_w8a8(flat,
                                                 self.smooth_factor,
                                                 self.proj_down if self.rank else None,
                                                 pad_rows=True)
    out = svdq_gemm_w8a8(qact,
                         self.qweight,
                         ascales,
                         self.wscales,
                         lora_act_in=lora_act if self.rank else None,
                         lora_up=self.proj_up if self.rank else None,
                         bias=self.bias)
    out = out[:rows, :self.out_features]
    return out.reshape(*shape[:-1], self.out_features)


def quantize_linear_svdq_fp8(
  linear: nn.Linear,
  representative_activations=None,
  *,
  rank: int = 32,
  alpha: float = 0.5,
  smooth_strategy: str = "identity",
  torch_dtype: torch.dtype | None = None,
  device: str | torch.device | None = None,
  calibrate_precision: str = "low",
) -> SVDQW8A8Linear:
  """Quantize a linear layer into SVDQ-FP8, with or without calibration data.

  :param linear: Source floating-point linear layer.
  :param representative_activations: Optional tensor or iterable of tensors used to derive the
    SmoothQuant factor when `smooth_strategy` is `activation`.
  :param rank: Low-rank correction rank (0, 32, 64 or 128).
  :param alpha: SmoothQuant interpolation factor.
  :param smooth_strategy: `activation`, `identity`, `weight` or `weight_inv`; `identity` disables
    smoothing, the two weight strategies derive it from the weight alone (DQ).
  :param torch_dtype: Dtype for the packed module's floating-point tensors.
  :param device: Target device for the packed module.
  :param calibrate_precision: Precision policy for the low-rank SVD.
  :returns: The packed `SVDQW8A8Linear`.
  """
  if not isinstance(linear, nn.Linear):
    raise TypeError(f"Expected nn.Linear, got {type(linear)}.")
  device = torch.device(device or linear.weight.device)
  torch_dtype = _normalize_dtype(torch_dtype, device)
  k = linear.in_features
  smooth = None
  if smooth_strategy == "activation":
    if representative_activations is None:
      raise ValueError("smooth_strategy='activation' requires representative activations.")
    acts = representative_activations
    if isinstance(acts, torch.Tensor):
      acts = [acts]
    span = None
    for tensor in acts:
      flat = tensor.detach().reshape(-1, tensor.shape[-1]).to(device=device, dtype=torch.float32)
      current = flat.abs().amax(dim=0)
      span = current if span is None else torch.maximum(span, current)
    weight_span = linear.weight.detach().to(device=device, dtype=torch.float32).abs().amax(dim=0)
    smooth = compute_smooth_scale(span,
                                  weight_span,
                                  alpha=alpha,
                                  math_dtype=torch.float32,
                                  output_dtype=torch.float32)
  elif smooth_strategy == "identity":
    smooth = torch.ones(k, dtype=torch.float32, device=device)
  elif smooth_strategy in ("weight", "weight_inv"):
    weight_span = linear.weight.detach().to(device=device, dtype=torch.float32).abs().amax(dim=0)
    smooth = _compute_weight_only_smooth_scale(weight_span,
                                               alpha=alpha,
                                               inverse=smooth_strategy == "weight_inv",
                                               math_dtype=torch.float32,
                                               output_dtype=torch.float32)
  else:
    raise ValueError(f"Unsupported SVDQ-FP8 smooth strategy: {smooth_strategy!r}.")
  return SVDQW8A8Linear.from_linear(linear,
                                    rank=rank,
                                    smooth_factor=smooth,
                                    alpha=alpha,
                                    torch_dtype=torch_dtype,
                                    device=device,
                                    calibrate_precision=calibrate_precision)


def quantize_svdq_fp8(module: nn.Module, quantize_config) -> nn.Module:
  """Quantize a module in place with SVDQ-FP8, honoring the shared SVDQ scoping rules.

  :param module: Root module containing the float `nn.Linear` layers to quantize.
  :param quantize_config: `QuantizeConfig` whose `quant_type` is `svdq_fp8_r{rank}[_dq]`.
  :returns: The module with eligible linears replaced by `SVDQW8A8Linear` instances.
  """
  if not isinstance(module, nn.Module):
    raise TypeError(f"Expected nn.Module, got {type(module)}.")
  if quantize_config.serialize_to is not None:
    raise NotImplementedError("SVDQ-FP8 does not support checkpoint serialization yet; leave "
                              "serialize_to=None. The W4A4 save/load format is not compatible "
                              "with the E4M3 residual layout.")
  from .ptq import SVDQPTQContext
  from .ptq import _call_with_supported_kwargs
  from .ptq import _get_named_submodule
  from .ptq import _get_parent_and_child
  from .ptq import _resolve_svdq_quantize_device

  context = SVDQPTQContext.from_config(module, quantize_config)
  kwargs = context.svdq_kwargs
  quantize_device = _resolve_svdq_quantize_device(module, kwargs)
  is_dq = quantize_config.is_svdq_dq()
  strategy = "activation" if not is_dq else kwargs.get("smooth_strategy", "identity")
  if strategy == "few_shot":
    raise ValueError("SVDQ-FP8 does not support the few_shot smooth strategy.")
  calibrate_precision = kwargs.get("calibrate_precision", "low")

  captured: dict[str, torch.Tensor] = {}
  if not is_dq:
    if quantize_config.calibrate_fn is None:
      raise ValueError("SVDQ-FP8 PTQ requires QuantizeConfig.calibrate_fn; use the `_dq` variant "
                       "for data-free quantization.")

    def make_hook(layer_name: str):

      def hook(_layer, args):
        if args and isinstance(args[0], torch.Tensor):
          value = args[0].detach()
          span = value.abs().reshape(-1, value.shape[-1]).amax(dim=0)
          captured[layer_name] = span if layer_name not in captured else torch.maximum(
            captured[layer_name], span)

      return hook

    handles = [
      _get_named_submodule(module, name).register_forward_pre_hook(make_hook(name))
      for name in context.candidate_layer_names
    ]
    try:
      _call_with_supported_kwargs(quantize_config.calibrate_fn, module=module, model=module)
    finally:
      for handle in handles:
        handle.remove()

  result = module
  for name in context.candidate_layer_names:
    linear = _get_named_submodule(module, name)
    packed = quantize_linear_svdq_fp8(linear,
                                      captured.get(name),
                                      rank=context.rank,
                                      smooth_strategy=strategy,
                                      torch_dtype=linear.weight.dtype,
                                      device=quantize_device,
                                      calibrate_precision=calibrate_precision)
    if name == "":
      result = packed
    else:
      parent, child = _get_parent_and_child(module, name)
      setattr(parent, child, packed)
  return result
