import logging
import os
from datetime import datetime

import torch
import torch.distributed as dist


def setup_logger(rank):
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}][%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def run(rank, world_size):
    setup_logger(rank)
    logger = logging.getLogger()
    try:
        # ================== 设备初始化 ==================
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            torch.cuda.set_device(rank % num_gpus)
            device = torch.device(f"cuda:{rank % num_gpus}")
            backend = "nccl" if num_gpus >= world_size else "gloo"
        else:
            device = torch.device("cpu")
            backend = "gloo"

        # ================== 进程组初始化 ==================
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        logger.info(f"Initialized with backend: {backend}, device: {device}")

        # ================== 数据准备 ==================
        # 基础张量数据（统一使用 float32）
        base_tensor = torch.tensor(
            [rank + 1.0, rank + 2.0, rank + 3.0],
            dtype=torch.float32,
            device=device,
        )
        obj_data = {"rank": rank, "time": datetime.now().strftime("%H:%M:%S")}

        # 预分配内存（保持数据类型一致）
        gather_tensor_list = [
            torch.zeros_like(base_tensor) for _ in range(world_size)
        ]
        gather_into_tensor = torch.zeros(
            world_size * 3, dtype=torch.float32, device=device
        )

        # 分发数据（显式指定数据类型）
        scatter_list = (
            [
                torch.tensor(
                    [i, i + 1.0, i + 2.0], dtype=torch.float32, device=device
                )
                for i in range(world_size)
            ]
            if rank == 0
            else None
        )
        scatter_obj_list = (
            [{"src": i, "data": f"data_{i}"} for i in range(world_size)]
            if rank == 0
            else None
        )

        # ================== 通信操作 ==================
        # 1. 广播张量
        dist.broadcast(base_tensor, src=0)
        logger.info(f"Broadcast result: {base_tensor.cpu()}")

        # 2. 广播对象列表
        obj_list = [obj_data] if rank == 0 else [None]
        dist.broadcast_object_list(obj_list, src=0)
        logger.info(f"Broadcast objects: {obj_list}")

        # 3. 全局归约 (求和)
        dist.all_reduce(base_tensor, op=dist.ReduceOp.SUM)
        logger.info(f"All_reduce SUM: {base_tensor.cpu()}")

        # 4. 目标归约到 rank 0
        if rank == 0:
            temp = base_tensor.clone()
        dist.reduce(base_tensor, dst=0, op=dist.ReduceOp.MAX)
        if rank == 0:
            logger.info(f"Reduce MAX result: {base_tensor.cpu()}")

        # 5. 全收集到列表
        dist.all_gather(gather_tensor_list, base_tensor)
        logger.info(
            f"All_gather results: {[t.cpu() for t in gather_tensor_list]}"
        )

        # 6. 全收集到大张量
        dist.all_gather_into_tensor(gather_into_tensor, base_tensor)
        logger.info(f"All_gather_into_tensor: {gather_into_tensor.cpu()}")

        # 7. 全收集对象
        all_objs = [{} for _ in range(world_size)]
        dist.all_gather_object(all_objs, obj_data)
        logger.info(f"All_gather_object: {all_objs}")

        # 8. 收集到 rank 0
        gather_dst_list = gather_tensor_list if rank == 0 else None
        dist.gather(base_tensor, gather_dst_list, dst=0)
        if rank == 0:
            logger.info(f"Gather results: {[t.cpu() for t in gather_dst_list]}")

        # 9. 收集对象到 rank 0
        obj_gather_list = all_objs if rank == 0 else None
        dist.gather_object(obj_data, obj_gather_list, dst=0)
        if rank == 0:
            logger.info(f"Gather_object results: {obj_gather_list}")

        # 10. 分发张量（已修复数据类型）
        recv_tensor = torch.zeros(3, dtype=torch.float32, device=device)
        dist.scatter(recv_tensor, scatter_list, src=0)
        logger.info(f"Scatter received: {recv_tensor.cpu()}")

        # 11. 分发对象
        recv_obj = [None]
        dist.scatter_object_list(recv_obj, scatter_obj_list, src=0)
        logger.info(f"Scatter_object received: {recv_obj}")

        # 12. Reduce-Scatter
        in_tensors = [torch.ones(3, dtype=torch.float32, device=device) * rank]
        out_tensor = torch.zeros(3, dtype=torch.float32, device=device)
        dist.reduce_scatter(out_tensor, in_tensors, op=dist.ReduceOp.SUM)
        logger.info(f"Reduce_scatter result: {out_tensor.cpu()}")

        # 13. Reduce-Scatter Tensor
        big_tensor = torch.cat(
            [torch.ones(3, dtype=torch.float32, device=device) * rank]
            * world_size
        )
        dist.reduce_scatter_tensor(
            out_tensor, big_tensor, op=dist.ReduceOp.PRODUCT
        )
        logger.info(f"Reduce_scatter_tensor result: {out_tensor.cpu()}")

        # 14. All-to-All Single
        in_tensor = torch.arange(
            3 * world_size, dtype=torch.float32, device=device
        ).view(-1)
        out_tensor = torch.zeros_like(in_tensor)
        dist.all_to_all_single(out_tensor, in_tensor)
        logger.info(f"All_to_all_single output: {out_tensor.cpu()}")

        # 15. All-to-All（已修复数据类型）
        in_list = [
            torch.tensor([rank * 10.0 + i], dtype=torch.float32, device=device)
            for i in range(world_size)
        ]
        out_list = [
            torch.zeros(1, dtype=torch.float32, device=device)
            for _ in range(world_size)
        ]
        dist.all_to_all(out_list, in_list)
        logger.info(f"All_to_all output: {[t.cpu() for t in out_list]}")

        # 16. 同步屏障
        dist.barrier()
        logger.info("Barrier passed")

        # 17. 异步操作
        async_tensor = base_tensor.clone()
        work = dist.all_reduce(
            async_tensor, op=dist.ReduceOp.MIN, async_op=True
        )
        work.wait()
        logger.info(f"Async op result: {async_tensor.cpu()}")

    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Process group destroyed")


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2

    if torch.cuda.is_available():
        visible_devices = ",".join(
            str(i) for i in range(torch.cuda.device_count())
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    r"""Spawns ``nprocs`` processes that run ``fn`` with ``args``.

    If one of the processes exits with a non-zero exit status, the
    remaining processes are killed and an exception is raised with the
    cause of termination. In the case an exception was caught in the
    child process, it is forwarded and its traceback is included in
    the exception raised in the parent process.

    Args:
        fn (function): Function is called as the entrypoint of the
            spawned process. This function must be defined at the top
            level of a module so it can be pickled and spawned. This
            is a requirement imposed by multiprocessing.

            The function is called as ``fn(i, *args)``, where ``i`` is
            the process index and ``args`` is the passed through tuple
            of arguments.

        args (tuple): Arguments passed to ``fn``.
        nprocs (int): Number of processes to spawn.
        join (bool): Perform a blocking join on all processes.
        daemon (bool): The spawned processes' daemon flag. If set to True,
                       daemonic processes will be created.
        start_method (str): (deprecated) this method will always use ``spawn``
                               as the start method. To use a different start method
                               use ``start_processes()``.

    Returns:
        None if ``join`` is ``True``,
        :class:`~ProcessContext` if ``join`` is ``False``

    """
    torch.multiprocessing.spawn(
        run, args=(world_size,), nprocs=world_size, join=True
    )
