import logging
import os

import ray
import torch
import torch.distributed as dist


def setup_logger(rank):
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Ray][Rank {rank}][%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )


@ray.remote(num_gpus=1)
class RayAll2AllWorker:
    def __init__(self, rank: int, world_size: int):
        assert world_size <= 4, "world_size must be <= 4"
        assert rank < world_size, "rank must be less than world_size"
        self.rank = rank
        self.world_size = world_size
        setup_logger(rank)
        self.logger = logging.getLogger()
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        if torch.cuda.is_available():
            self.device = f"cuda:{torch.cuda.current_device()}"
            self.backend = "nccl"
            self.logger.info(f"Using GPU: {self.device}")
        else:
            self.device = torch.device("cpu")
            self.backend = "gloo"

        dist.init_process_group(
            backend=self.backend,
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
        )
        self.logger.info(
            f"Initialized with backend: {self.backend}, device: {self.device}"
        )

    def run(self) -> torch.Tensor:

        try:
            # All-to-All Single
            input_t = torch.arange(4, device=self.device) + self.rank * 4
            output_t = torch.empty([4], dtype=torch.int64, device=self.device)
            dist.all_to_all_single(output_t, input_t)
            self.logger.info(
                f"All-to-All Single output_t at Rank {self.rank}:\n {output_t}"
            )

            # All-to-All Single UnEven
            if self.rank == 0:
                input_t = torch.tensor(
                    [0, 1, 2, 3, 4, 5], dtype=torch.int64, device=self.device
                )
            elif self.rank == 1:
                input_t = torch.tensor(
                    [10, 11, 12, 13, 14, 15, 16, 17, 18],
                    dtype=torch.int64,
                    device=self.device,
                )
            elif self.rank == 2:
                input_t = torch.tensor(
                    [20, 21, 22, 23, 24], dtype=torch.int64, device=self.device
                )
            elif self.rank == 3:
                input_t = torch.tensor(
                    [30, 31, 32, 33, 34, 35, 36],
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                input_t = None

            assert input_t is not None, "Input tensor should not be None"

            input_splits = [
                [2, 2, 1, 1],
                [3, 2, 2, 2],
                [2, 1, 1, 1],
                [2, 2, 2, 1],
            ]

            output_splits = [
                [2, 3, 2, 2],
                [2, 2, 1, 2],
                [1, 2, 1, 2],
                [1, 2, 1, 1],
            ]

            output_t = torch.empty(
                sum(output_splits[self.rank]),
                dtype=torch.int64,
                device=self.device,
            )
            dist.all_to_all_single(
                output_t,
                input_t,
                output_split_sizes=output_splits[self.rank],
                input_split_sizes=input_splits[self.rank],
            )
            self.logger.info(
                f"All-to-All Single <UnEven> output_t at Rank {self.rank}:\n {output_t}"
            )
            return output_t

        except Exception as e:
            self.logger.error(f"Process failed: {str(e)}")
            raise e


if __name__ == "__main__":
    # export RAY_DEDUP_LOGS=0
    world_size = torch.cuda.device_count()
    print(f"world_size: {world_size}")
    if not ray.is_initialized():
        ray.init()
    workers = [
        RayAll2AllWorker.remote(rank, world_size) for rank in range(world_size)
    ]
    results = ray.get([worker.run.remote() for worker in workers])
    ray.shutdown()
    print("All processes completed successfully")
