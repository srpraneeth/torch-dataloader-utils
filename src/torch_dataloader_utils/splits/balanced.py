import logging
import random

from torch_dataloader_utils.splits.core import DataFileInfo, Shard, Split

logger = logging.getLogger(__name__)


def _file_weight(file: DataFileInfo) -> int | None:
    """Returns the best available weight — record_count preferred, then file_size."""
    if file.record_count is not None:
        return file.record_count
    if file.file_size is not None:
        return file.file_size
    return None


class SizeBalancedSplitStrategy:
    """Distributes files across workers balancing by data volume.

    Uses record_count when available (Iceberg), falls back to file_size
    (plain files), falls back to round-robin when neither is available.
    Uses a greedy bin-packing algorithm — assigns each file to the shard
    with the current lowest total weight.
    All files are read in full (row_range=None).
    Satisfies the SplitStrategy protocol.
    """

    def __init__(self, shuffle: bool = False, seed: int = 42) -> None:
        self.shuffle = shuffle
        self.seed = seed

    def generate(
        self,
        files: list[DataFileInfo],
        num_workers: int,
        epoch: int = 0,
    ) -> list[Shard]:
        file_list = list(files)

        if self.shuffle:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(file_list)

        weights = [_file_weight(f) for f in file_list]

        # fall back to round-robin if no weight metadata available
        if all(w is None for w in weights):
            logger.info(
                "SizeBalancedSplitStrategy: no weight metadata, falling back to round-robin  "
                "files=%d workers=%d epoch=%d",
                len(files),
                num_workers,
                epoch,
            )
            shards = [Shard(id=i) for i in range(num_workers)]
            for i, file in enumerate(file_list):
                shards[i % num_workers].splits.append(Split(file=file))
            return shards

        # replace missing weights with 0 for bin-packing
        weights = [w if w is not None else 0 for w in weights]
        weight_metric = (
            "record_count" if file_list and file_list[0].record_count is not None else "file_size"
        )

        # greedy bin-packing — assign each file to the least-loaded shard
        shards = [Shard(id=i) for i in range(num_workers)]
        totals = [0] * num_workers

        for file, weight in sorted(zip(file_list, weights), key=lambda x: x[1], reverse=True):
            min_idx = totals.index(min(totals))
            shards[min_idx].splits.append(Split(file=file))
            totals[min_idx] += weight

        logger.info(
            "SizeBalancedSplitStrategy: %d files → %d workers  epoch=%d  shuffle=%s  "
            "metric=%s  shard_totals=%s",
            len(files),
            num_workers,
            epoch,
            self.shuffle,
            weight_metric,
            totals,
        )
        for shard in shards:
            shard_weight = sum(_file_weight(s.file) or 0 for s in shard.splits)
            logger.debug(
                "Shard %d: %d split(s)  total_%s=%d  files=%s",
                shard.id,
                len(shard.splits),
                weight_metric,
                shard_weight,
                [s.file.path for s in shard.splits],
            )
        return shards
