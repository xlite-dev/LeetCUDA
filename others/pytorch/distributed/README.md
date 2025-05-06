# Learning PyTorch Distributed Communication

## Official Docs

[Distributed communication package - torch.distributed](https://pytorch.org/docs/stable/distributed.html)


## All in One Test

```bash
python3 test_dist_all.py

# log
[Rank 1][16:09:12] Initialized with backend: nccl, device: cuda:1
[Rank 0][16:09:12] Initialized with backend: nccl, device: cuda:0
[Rank 1][16:09:13] Broadcast result: tensor([1., 2., 3.])
[Rank 0][16:09:13] Broadcast result: tensor([1., 2., 3.])
[Rank 0][16:09:13] Broadcast objects: [{'rank': 0, 'time': '16:09:12'}]
[Rank 1][16:09:13] Broadcast objects: [{'rank': 0, 'time': '16:09:12'}]
[Rank 0][16:09:13] All_reduce SUM: tensor([2., 4., 6.])
[Rank 1][16:09:13] All_reduce SUM: tensor([2., 4., 6.])
[Rank 0][16:09:13] Reduce MAX result: tensor([2., 4., 6.])
[Rank 1][16:09:13] All_gather results: [tensor([2., 4., 6.]), tensor([2., 4., 6.])]
[Rank 0][16:09:13] All_gather results: [tensor([2., 4., 6.]), tensor([2., 4., 6.])]
[Rank 1][16:09:13] All_gather_into_tensor: tensor([2., 4., 6., 2., 4., 6.])
[Rank 0][16:09:13] All_gather_into_tensor: tensor([2., 4., 6., 2., 4., 6.])
[Rank 1][16:09:13] All_gather_object: [{'rank': 0, 'time': '16:09:12'}, {'rank': 1, 'time': '16:09:12'}]
[Rank 0][16:09:13] All_gather_object: [{'rank': 0, 'time': '16:09:12'}, {'rank': 1, 'time': '16:09:12'}]
[Rank 0][16:09:13] Gather results: [tensor([2., 4., 6.]), tensor([2., 4., 6.])]
[Rank 0][16:09:13] Gather_object results: [{'rank': 0, 'time': '16:09:12'}, {'rank': 1, 'time': '16:09:12'}]
[Rank 0][16:09:13] Scatter received: tensor([0., 1., 2.])
[Rank 1][16:09:13] Scatter received: tensor([1., 2., 3.])
[Rank 0][16:09:13] Scatter_object received: [{'src': 0, 'data': 'data_0'}]
[Rank 1][16:09:13] Scatter_object received: [{'src': 1, 'data': 'data_1'}]
[Rank 1][16:09:13] Reduce_scatter result: tensor([ 8.2378e-37,  2.8818e+26, -9.8832e-32])
[Rank 0][16:09:13] Reduce_scatter result: tensor([1., 1., 1.])
[Rank 0][16:09:13] Reduce_scatter_tensor result: tensor([0., 0., 0.])
[Rank 1][16:09:13] Reduce_scatter_tensor result: tensor([0., 0., 0.])
[Rank 0][16:09:13] All_to_all_single output: tensor([0., 1., 2., 0., 1., 2.])
[Rank 1][16:09:13] All_to_all_single output: tensor([3., 4., 5., 3., 4., 5.])
[Rank 0][16:09:13] All_to_all output: [tensor([0.]), tensor([10.])]
[Rank 1][16:09:13] All_to_all output: [tensor([1.]), tensor([11.])]
[Rank 0][16:09:13] Barrier passed
[Rank 1][16:09:13] Barrier passed
[Rank 0][16:09:13] Async op result: tensor([2., 4., 6.])
[Rank 1][16:09:13] Async op result: tensor([2., 4., 6.])
[Rank 0][16:09:13] Process group destroyed
[Rank 1][16:09:13] Process group destroyed
```

## All-to-All

```bash
python3 test_all_to_all.py
python3 test_all_to_all_single.py
```

## TODO
