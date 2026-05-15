"""Manual sharding baseline: files are pre-partitioned across workers at startup.

Each worker reads only its assigned files (every Nth file starting from wid).
I/O amplification = 1× — no data is read twice. This represents what a careful
engineer writes without this library. It matches our throughput on equal-sized
files but degrades on unequal files (S2) and cannot parallelise a single large
file across workers (S3).
"""

from __future__ import annotations

from glob import glob
from typing import Iterator

import pyarrow.parquet as pq
import torch.utils.data as tud

from benchmarks._common import passthrough


class ManualShardedDataset(tud.IterableDataset):
    def __init__(self, data_dir: str, batch_size: int = 1024):
        self._files = sorted(glob(f"{data_dir}/*.parquet"))
        self._batch_size = batch_size

    def __iter__(self) -> Iterator:
        info = tud.get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        my_files = self._files[wid::nw]
        for path in my_files:
            pf = pq.ParquetFile(path)
            yield from pf.iter_batches(self._batch_size)


def make_loader(data_dir: str, num_workers: int, batch_size: int = 1024) -> tud.DataLoader:
    dataset = ManualShardedDataset(data_dir, batch_size=batch_size)
    return tud.DataLoader(dataset, batch_size=None, num_workers=num_workers, collate_fn=passthrough)
