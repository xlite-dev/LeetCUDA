"""SVDQ-FP8 (W8A8) operators: fused activation/LoRA-down producer and fused GEMM.

Numerical contract
==================
For a Linear with weight ``W[N,K]`` and smoothing ``s[K]``, the offline quantizer writes
``W*s = L1 @ L2 + R`` and stores the E4M3 residual plus its per-output-channel scale::

    qB[n,k] = satfinite_e4m3(R[n,k] / sb[n]),  sb[n] = amax_k |R[n,:]| / 448

The producer consumes the *raw* activation ``X[M,K]`` and emits, in one kernel::

    sa[m]   = max(amax_k |X/s|, eps) / 448
    qA[m,k] = satfinite_e4m3((X/s) / sa[m])
    D[m,r]  = bf16(sum_k X[m,k] * proj_down[r,k])       (FP32 accumulation, one rounding)

``proj_down`` is ``L2 / s``, so the low-rank branch reads the unsmoothed activation exactly like
the NVFP4 path. The GEMM then evaluates, with a single FP32 accumulator for both products::

    out = bf16((qA @ qB.T) * (sa[m] * sb[n]) + bias[n] + D @ L1.T)

Neither scale touches the LoRA term, and no standalone BF16 GEMM or second pass over the [M,N]
output exists anywhere in the chain.
"""

import torch

from .cuda_ext import _get_required_ops_module

_SUPPORTED_RANKS = (0, 32, 64, 128)


def _call_svdq_gemm_w8a8(act: torch.Tensor, wgt: torch.Tensor, out: torch.Tensor,
                         ascales: torch.Tensor, wscales: torch.Tensor,
                         lora_act_in: torch.Tensor | None, lora_up: torch.Tensor | None,
                         bias: torch.Tensor | None) -> None:
  """Invoke the pybind W8A8 GEMM with the caller-allocated output buffer."""
  ops = _get_required_ops_module()
  ops.gemm_w8a8(act=act,
                wgt=wgt,
                out=out,
                ascales=ascales,
                wscales=wscales,
                lora_act_in=lora_act_in,
                lora_up=lora_up,
                bias=bias)


torch.library.define(
  "cache_dit_cuda_ops::svdq_gemm_w8a8",
  "(Tensor act, Tensor wgt, Tensor ascales, Tensor wscales, Tensor? lora_act_in, Tensor? lora_up, "
  "Tensor? bias) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::svdq_gemm_w8a8", "CUDA")
def _svdq_gemm_w8a8(act: torch.Tensor, wgt: torch.Tensor, ascales: torch.Tensor,
                    wscales: torch.Tensor, lora_act_in: torch.Tensor | None,
                    lora_up: torch.Tensor | None, bias: torch.Tensor | None) -> torch.Tensor:
  m, _ = act.shape
  n = wgt.shape[0]
  out = torch.empty((m, n), dtype=torch.bfloat16, device=act.device)
  _call_svdq_gemm_w8a8(act, wgt, out, ascales, wscales, lora_act_in, lora_up, bias)
  return out


@torch.library.register_fake("cache_dit_cuda_ops::svdq_gemm_w8a8")
def _fake_svdq_gemm_w8a8(act: torch.Tensor, wgt: torch.Tensor, ascales: torch.Tensor,
                         wscales: torch.Tensor, lora_act_in: torch.Tensor | None,
                         lora_up: torch.Tensor | None, bias: torch.Tensor | None) -> torch.Tensor:
  del ascales, wscales, lora_act_in, lora_up, bias
  return act.new_empty((act.shape[0], wgt.shape[0]), dtype=torch.bfloat16)


def svdq_gemm_w8a8(act: torch.Tensor,
                   wgt: torch.Tensor,
                   ascales: torch.Tensor,
                   wscales: torch.Tensor,
                   lora_act_in: torch.Tensor | None = None,
                   lora_up: torch.Tensor | None = None,
                   bias: torch.Tensor | None = None) -> torch.Tensor:
  """Compute the fused SVDQ-FP8 linear from E4M3 operands and a BF16 low-rank branch.

  :param act: E4M3 activation codes ``[M,K]``, contiguous.
  :param wgt: E4M3 residual codes ``[N,K]``, contiguous.
  :param ascales: FP32 per-token activation scales ``[M]``.
  :param wscales: FP32 per-output-channel residual scales ``[N]``.
  :param lora_act_in: Optional BF16 LoRA-down output ``[M,R]``.
  :param lora_up: Optional BF16 LoRA-up weight ``[N,R]``.
  :param bias: Optional BF16 bias ``[N]``.
  :returns: BF16 output ``[M,N]``.
  """
  if act.dtype != torch.float8_e4m3fn or wgt.dtype != torch.float8_e4m3fn:
    raise ValueError("svdq_gemm_w8a8 expects float8_e4m3fn activation and weight codes.")
  if act.dim() != 2 or wgt.dim() != 2:
    raise ValueError("svdq_gemm_w8a8 expects 2D activation and weight tensors.")
  if not act.is_contiguous() or not wgt.is_contiguous():
    raise ValueError("svdq_gemm_w8a8 expects contiguous activation and weight tensors.")
  m, k = act.shape
  n = wgt.shape[0]
  if wgt.shape[1] != k:
    raise ValueError("svdq_gemm_w8a8 inner dimensions do not match.")
  if m % 128 != 0 or n % 128 != 0 or k % 128 != 0:
    raise ValueError(f"svdq_gemm_w8a8 requires M/N/K multiples of 128, got {(m, n, k)}.")
  if ascales.dtype != torch.float32 or wscales.dtype != torch.float32:
    raise ValueError("svdq_gemm_w8a8 expects FP32 scales.")
  if ascales.numel() != m or wscales.numel() != n:
    raise ValueError("svdq_gemm_w8a8 scale shapes do not match the operands.")
  if (lora_act_in is None) != (lora_up is None):
    raise ValueError("svdq_gemm_w8a8 requires both LoRA tensors or neither.")
  if lora_up is not None:
    rank = lora_up.shape[1]
    if rank not in _SUPPORTED_RANKS or rank == 0:
      raise ValueError(f"svdq_gemm_w8a8 supports rank 32/64/128, got {rank}.")
    if lora_act_in.dtype != torch.bfloat16 or lora_up.dtype != torch.bfloat16:
      raise ValueError("svdq_gemm_w8a8 expects BF16 LoRA tensors.")
    if tuple(lora_act_in.shape) != (m, rank) or tuple(lora_up.shape) != (n, rank):
      raise ValueError("svdq_gemm_w8a8 LoRA shapes do not match the operands.")
    if not lora_act_in.is_contiguous() or not lora_up.is_contiguous():
      raise ValueError("svdq_gemm_w8a8 expects contiguous LoRA tensors.")
  if bias is not None and (bias.dtype != torch.bfloat16 or bias.numel() != n):
    raise ValueError("svdq_gemm_w8a8 expects a BF16 bias of length N.")
  return torch.ops.cache_dit_cuda_ops.svdq_gemm_w8a8(act=act,
                                                     wgt=wgt,
                                                     ascales=ascales,
                                                     wscales=wscales,
                                                     lora_act_in=lora_act_in,
                                                     lora_up=lora_up,
                                                     bias=bias)


def _call_svdq_quantize_w8a8(input: torch.Tensor, output: torch.Tensor, ascales: torch.Tensor,
                             lora_down: torch.Tensor | None, lora_act: torch.Tensor | None,
                             smooth: torch.Tensor) -> None:
  """Invoke the pybind fused producer with caller-allocated outputs."""
  ops = _get_required_ops_module()
  ops.quantize_w8a8(input=input,
                    output=output,
                    ascales=ascales,
                    lora_down=lora_down,
                    lora_act=lora_act,
                    smooth=smooth)


torch.library.define(
  "cache_dit_cuda_ops::svdq_quantize_w8a8",
  "(Tensor input, Tensor smooth, Tensor? lora_down) -> (Tensor output, Tensor ascales, "
  "Tensor lora_act)",
)


def _w8a8_padded_rows(rows: int) -> int:
  return (rows + 127) // 128 * 128


@torch.library.impl("cache_dit_cuda_ops::svdq_quantize_w8a8", "CUDA")
def _svdq_quantize_w8a8(
    input: torch.Tensor, smooth: torch.Tensor,
    lora_down: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  rows, channels = input.shape
  rows_pad = _w8a8_padded_rows(rows)
  rank = 0 if lora_down is None else lora_down.shape[0]
  padded = input.contiguous()
  if rows_pad != rows:
    padded = torch.nn.functional.pad(padded, (0, 0, 0, rows_pad - rows))
  output = torch.empty((rows_pad, channels), dtype=torch.float8_e4m3fn, device=input.device)
  ascales = torch.empty(rows_pad, dtype=torch.float32, device=input.device)
  lora_act = torch.empty((rows_pad, rank), dtype=torch.bfloat16, device=input.device)
  _call_svdq_quantize_w8a8(padded, output, ascales, lora_down, lora_act if rank else None, smooth)
  return output, ascales, lora_act


@torch.library.register_fake("cache_dit_cuda_ops::svdq_quantize_w8a8")
def _fake_svdq_quantize_w8a8(
    input: torch.Tensor, smooth: torch.Tensor,
    lora_down: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  del smooth
  rows, channels = input.shape
  rows_pad = _w8a8_padded_rows(rows)
  rank = 0 if lora_down is None else lora_down.shape[0]
  return (input.new_empty((rows_pad, channels),
                          dtype=torch.float8_e4m3fn), input.new_empty(rows_pad,
                                                                      dtype=torch.float32),
          input.new_empty((rows_pad, rank), dtype=torch.bfloat16))


def svdq_quantize_w8a8(
    input: torch.Tensor,
    smooth: torch.Tensor,
    lora_down: torch.Tensor | None = None,
    pad_rows: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
  """Quantize an activation and compute the LoRA-down term in one fused kernel.

  :param input: BF16 activation ``[M,K]``.
  :param smooth: BF16 per-input-channel divisor ``[K]``.
  :param lora_down: Optional BF16 ``[R,K]`` down projection, already divided by ``smooth``.
  :param pad_rows: If `True`, return the 128-aligned padded outputs required by `svdq_gemm_w8a8`;
    otherwise slice rows back to M. Use `True` when composing producer and GEMM with ragged M.
  :returns: ``(qact e4m3, ascales fp32, lora_act bf16 or None)`` with M or padded rows.
  """
  if input.dtype != torch.bfloat16 or input.dim() != 2:
    raise ValueError("svdq_quantize_w8a8 expects a 2D BF16 activation.")
  if smooth.dtype != torch.bfloat16 or smooth.numel() != input.shape[1]:
    raise ValueError("svdq_quantize_w8a8 expects a BF16 smooth vector of length K.")
  if input.shape[1] % 64 != 0:
    raise ValueError(f"svdq_quantize_w8a8 requires K divisible by 64, got {input.shape[1]}.")
  if lora_down is not None:
    if lora_down.dtype != torch.bfloat16 or lora_down.dim() != 2:
      raise ValueError("svdq_quantize_w8a8 expects a 2D BF16 lora_down.")
    if lora_down.shape[0] not in _SUPPORTED_RANKS or lora_down.shape[0] == 0:
      raise ValueError(f"svdq_quantize_w8a8 supports rank 32/64/128, got {lora_down.shape[0]}.")
    if lora_down.shape[1] != input.shape[1] or not lora_down.is_contiguous():
      raise ValueError("svdq_quantize_w8a8 expects a contiguous lora_down of shape [R,K].")
  rows = input.shape[0]
  if rows == 0:
    rank = 0 if lora_down is None else lora_down.shape[0]
    return (input.new_empty((0, input.shape[1]), dtype=torch.float8_e4m3fn),
            input.new_empty(0, dtype=torch.float32), input.new_empty(
              (0, rank), dtype=torch.bfloat16) if rank else None)
  output, ascales, lora_act = torch.ops.cache_dit_cuda_ops.svdq_quantize_w8a8(input=input,
                                                                              smooth=smooth,
                                                                              lora_down=lora_down)
  if pad_rows:
    return output, ascales, (lora_act if lora_down is not None else None)
  return output[:rows], ascales[:rows], (lora_act[:rows] if lora_down is not None else None)


__all__ = [
  "svdq_gemm_w8a8",
  "svdq_quantize_w8a8",
]
