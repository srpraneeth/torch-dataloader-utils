import logging
import random

from torch_dataloader_utils.splits.core import DataFileInfo, Shard, Split

logger = logging.getLogger(__name__)


class RoundRobinSplitStrategy:
    """Distributes files across workers using round-robin assignment.

    Best for equi-sized files where file count is a good proxy for data volume.
    Ignores file_size and record_count metadata.
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

        shards = [Shard(id=i) for i in range(num_workers)]
        for i, file in enumerate(file_list):
            shards[i % num_workers].splits.append(Split(file=file))

        split_counts = [len(s.splits) for s in shards]
        logger.info(
            "RoundRobinSplitStrategy: %d files → %d workers  epoch=%d  shuffle=%s  "
            "splits_per_worker=%s",
            len(files), num_workers, epoch, self.shuffle,
            split_counts,
        )
        for shard in shards:
            paths = [s.file.path for s in shard.splits]
            logger.debug(
                "Shard %d: %d split(s)  %s",
                shard.id, len(paths), paths,
            )
        return shards
