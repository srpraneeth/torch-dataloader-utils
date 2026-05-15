"""Naive baseline: every worker reads every file and filters batches by index parity.

This is the anti-pattern this library exists to replace. With num_workers=N,
each worker reads all files and discards (N-1)/N of the data — I/O amplification
of exactly num_workers×. Throughput (rows/sec) degrades because workers spend
most of their time reading data they throw away.
"""

from __future__ import annotations

from glob import glob
from typing import Iterator

import pyarrow.parquet as pq
import torch.utils.data as tud

from benchmarks._common import passthrough


class NaiveIterableDataset(tud.IterableDataset):
    def __init__(self, data_dir: str, batch_size: int = 1024):
        self._files = sorted(glob(f"{data_dir}/*.parquet"))
        self._batch_size = batch_size

    def __iter__(self) -> Iterator:
        info = tud.get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        for path in self._files:
            pf = pq.ParquetFile(path)
            for i, batch in enumerate(pf.iter_batches(self._batch_size)):
                if i % nw == wid:
                    yield batch


def make_loader(data_dir: str, num_workers: int, batch_size: int = 1024) -> tud.DataLoader:
    dataset = NaiveIterableDataset(data_dir, batch_size=batch_size)
    return tud.DataLoader(dataset, batch_size=None, num_workers=num_workers, collate_fn=passthrough)
