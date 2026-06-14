"""Single-pass indexing of MITgcm run directories (.meta filenames only)."""

from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field

from .meta import MetaSummary, read_meta_summary

_ITER_META_RE = re.compile(r"^(.+)\.(\d{10})(?:\.\d{3}\.\d{3})?$")

_catalog_cache: dict[str, RunCatalog] = {}


@dataclass
class RunCatalog:
    """Prefix list, iteration numbers, and lazy shape metadata for one run folder."""

    data_dir: str
    prefixes: list[str]
    iterations: dict[str, list[int]]
    _sample_meta: dict[str, str] = field(default_factory=dict)
    _shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    _meta_summaries: dict[str, MetaSummary] = field(default_factory=dict)

    @classmethod
    def scan(cls, data_dir: str) -> RunCatalog:
        data_dir = os.path.abspath(data_dir)
        iters_map: dict[str, list[int]] = defaultdict(list)
        best_iter: dict[str, int | None] = {}
        sample_meta: dict[str, str] = {}

        for meta_path in glob.glob(os.path.join(data_dir, "*.meta")):
            base = os.path.basename(meta_path)[:-5]
            match = _ITER_META_RE.match(base)
            if match:
                prefix, itr_str = match.group(1), match.group(2)
                itr = int(itr_str)
                iters_map[prefix].append(itr)
                prev = best_iter.get(prefix)
                if prev is None or itr >= prev:
                    best_iter[prefix] = itr
                    sample_meta[prefix] = meta_path
            else:
                iters_map[base]
                sample_meta[base] = meta_path
                best_iter.setdefault(base, None)

        for prefix, itr_list in iters_map.items():
            if itr_list:
                itr_list.sort()

        prefixes = sorted(iters_map)
        return cls(
            data_dir=data_dir,
            prefixes=prefixes,
            iterations=dict(iters_map),
            _sample_meta=sample_meta,
        )

    def iters_for(self, prefix: str) -> list[int]:
        return self.iterations.get(prefix, [])

    def iter_count(self, prefix: str) -> int:
        return len(self.iterations.get(prefix, []))

    def meta_summary(self, prefix: str) -> MetaSummary:
        if prefix not in self._meta_summaries:
            itr_list = self.iterations.get(prefix, [])
            sample_iter = itr_list[-1] if itr_list else None
            self._meta_summaries[prefix] = read_meta_summary(
                self.data_dir, prefix, sample_iter
            )
        return self._meta_summaries[prefix]

    def shape(self, prefix: str) -> tuple[int, ...]:
        if prefix not in self._shapes:
            self._shapes[prefix] = self.meta_summary(prefix).shape
        return self._shapes[prefix]


def get_catalog(data_dir: str, *, refresh: bool = False) -> RunCatalog:
    data_dir = os.path.abspath(data_dir)
    if refresh:
        _catalog_cache.pop(data_dir, None)
    if data_dir not in _catalog_cache:
        _catalog_cache[data_dir] = RunCatalog.scan(data_dir)
    return _catalog_cache[data_dir]


def invalidate_catalog(data_dir: str | None = None) -> None:
    if data_dir is None:
        _catalog_cache.clear()
    else:
        _catalog_cache.pop(os.path.abspath(data_dir), None)
