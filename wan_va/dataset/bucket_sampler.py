from collections import defaultdict
from math import ceil

import torch
from torch.utils.data import Sampler


class BucketedDistributedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        bucket_keys,
        batch_size,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=42,
        pad_batches=True,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_replicas <= 0:
            raise ValueError(
                f"num_replicas must be positive, got {num_replicas}"
            )
        if not 0 <= rank < num_replicas:
            raise ValueError(
                f"rank must be in [0, {num_replicas}), got {rank}"
            )

        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.pad_batches = pad_batches
        self.epoch = 0
        self.global_batch_size = batch_size * num_replicas

        bucket_to_indices = defaultdict(list)
        for index, key in enumerate(bucket_keys):
            bucket_to_indices[key].append(index)
        self.bucket_to_indices = {
            key: indices for key, indices in bucket_to_indices.items() if indices
        }
        self._length = self._compute_length()

    def _compute_length(self):
        total = 0
        for indices in self.bucket_to_indices.values():
            bucket_size = len(indices)
            if self.pad_batches:
                total += ceil(bucket_size / self.global_batch_size)
            else:
                total += bucket_size // self.global_batch_size
        return total

    def __len__(self):
        return self._length

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        global_batches = []
        for key in sorted(self.bucket_to_indices):
            indices = list(self.bucket_to_indices[key])
            if self.shuffle:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in order]

            remainder = len(indices) % self.global_batch_size
            if remainder:
                if not self.pad_batches:
                    indices = indices[: len(indices) - remainder]
                else:
                    needed = self.global_batch_size - remainder
                    # Repeat shuffled samples from the same bucket so every
                    # local batch keeps a consistent temporal shape.
                    indices.extend(indices[:needed])

            for start in range(0, len(indices), self.global_batch_size):
                batch = indices[start : start + self.global_batch_size]
                if len(batch) == self.global_batch_size:
                    global_batches.append(batch)

        if self.shuffle and global_batches:
            order = torch.randperm(len(global_batches), generator=generator).tolist()
            global_batches = [global_batches[i] for i in order]

        rank_offset = self.rank * self.batch_size
        local_batches = [
            batch[rank_offset : rank_offset + self.batch_size]
            for batch in global_batches
        ]
        return iter(local_batches)
