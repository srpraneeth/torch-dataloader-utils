import logging
import random

from torch_dataloader_utils.splits.core import DataFileInfo, FileSplit, Split

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
    ) -> list[Split]:
        file_list = list(files)

        if self.shuffle:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(file_list)

        splits = [Split(id=i) for i in range(num_workers)]
        for i, file in enumerate(file_list):
            splits[i % num_workers].file_splits.append(FileSplit(file=file))

        logger.debug(
            "RoundRobin splits: %d files → %d workers, epoch=%d, shuffle=%s",
            len(files), num_workers, epoch, self.shuffle,
        )
        return splits
