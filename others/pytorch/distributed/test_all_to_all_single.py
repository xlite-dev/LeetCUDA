import logging
import os

import torch
import torch.distributed as dist


def setup_logger(rank):
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}][%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def run(rank, world_size):
    assert world_size == 4, "world_size must be 4"
    setup_logger(rank)
    logger = logging.getLogger()
    try:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            torch.cuda.set_device(rank % num_gpus)
            device = torch.device(f"cuda:{rank % num_gpus}")
            backend = "nccl" if num_gpus >= world_size else "gloo"
        else:
            device = torch.device("cpu")
            backend = "gloo"

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        logger.info(f"Initialized with backend: {backend}, device: {device}")

        # All-to-All Single
        input = torch.arange(4, device=device) + rank * 4
        output = torch.empty([4], dtype=torch.int64, device=device)
        dist.all_to_all_single(output, input)
        logger.info(f"All-to-All Single output at Rank {rank}:\n {output}")

        # All-to-All Single UnEven
        if rank == 0:
            input = torch.tensor(
                [0, 1, 2, 3, 4, 5], dtype=torch.int64, device=device
            )
        elif rank == 1:
            input = torch.tensor(
                [10, 11, 12, 13, 14, 15, 16, 17, 18],
                dtype=torch.int64,
                device=device,
            )
        elif rank == 2:
            input = torch.tensor(
                [20, 21, 22, 23, 24], dtype=torch.int64, device=device
            )
        elif rank == 3:
            input = torch.tensor(
                [30, 31, 32, 33, 34, 35, 36], dtype=torch.int64, device=device
            )
        else:
            input = None

        assert input is not None, "Input tensor should not be None"

        input_splits = [[2, 2, 1, 1], [3, 2, 2, 2], [2, 1, 1, 1], [2, 2, 2, 1]]

        output_splits = [[2, 3, 2, 2], [2, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 1]]

        output = torch.empty(
            sum(output_splits[rank]), dtype=torch.int64, device=device
        )
        dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_splits[rank],
            input_split_sizes=input_splits[rank],
        )
        logger.info(
            f"All-to-All Single <UnEven> output at Rank {rank}:\n {output}"
        )

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
    world_size = 4

    if torch.cuda.is_available():
        visible_devices = ",".join(
            str(i) for i in range(torch.cuda.device_count())
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        assert torch.cuda.device_count() >= world_size, (
            f"CUDA_VISIBLE_DEVICES={visible_devices} "
            f"but world_size={world_size}"
        )

    torch.multiprocessing.spawn(
        run, args=(world_size,), nprocs=world_size, join=True
    )
