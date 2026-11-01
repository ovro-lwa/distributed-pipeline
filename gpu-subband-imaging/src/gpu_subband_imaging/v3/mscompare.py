"""Structure and content comparison of two casacore tables (main table and subtables).

Used by the v3 prep tests and scripts/v3_prep_compare.py to check that
``fastprep.write_ms_like`` writes what CASA mstransform writes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence

import numpy as np


def _eq(a: Any, b: Any) -> bool:
    """Exact equality; float arrays compared by their bytes."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.dtype.kind in "fc" and a.size:
            return np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
        return bool(np.array_equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return a == b


def _no_links(kw: dict) -> dict:
    # subtable links ("Table: /abs/path/SUB") differ by location only
    return {k: v for k, v in kw.items() if not (isinstance(v, str) and v.startswith("Table: "))}


def _desc(t) -> dict:
    d = t.getdesc()
    if "_keywords_" in d:
        d["_keywords_"] = _no_links(d["_keywords_"])
    return d


def _dminfo(t) -> dict:
    return {v["NAME"]: (v["TYPE"], sorted(v["COLUMNS"]), v.get("SPEC", {}))
            for v in t.getdminfo().values()}


def _column_values(t, col: str):
    """getcol, or (undefined or variable-shape cells) a per-cell list, None if undefined."""
    try:
        return t.getcol(col)
    except RuntimeError:
        pass
    return [t.getcell(col, r) if t.iscelldefined(col, r) else None for r in range(t.nrows())]


def subtables(ms: str) -> List[str]:
    return sorted(p.name for p in Path(ms).iterdir() if (p / "table.dat").is_file())


def table_diff(pa: str, pb: str, values: bool = True, skip_cols: Sequence[str] = (),
               with_subtables: bool = True, prefix: str = "") -> List[str]:
    """Human-readable differences between two tables; empty when they are the same.

    Compared: row count, column descriptions (incl. column keywords and storage
    manager groups), table keywords (apart from subtable paths), data manager types,
    groups and specs (tile shapes, bucket sizes), table info, and with ``values`` every
    column's contents (undefined cells must match).  With ``with_subtables`` every
    subtable is compared the same way.
    """
    from casacore.tables import table
    diffs: List[str] = []
    with table(str(pa), ack=False) as a, table(str(pb), ack=False) as b:
        if a.nrows() != b.nrows():
            diffs.append(f"{prefix}nrows {a.nrows()} != {b.nrows()}")
        da, db = _desc(a), _desc(b)
        if sorted(da) != sorted(db):
            diffs.append(f"{prefix}columns differ: {sorted(set(da) ^ set(db))}")
        for c in sorted(set(da) & set(db)):
            if not _eq(da[c], db[c]):
                diffs.append(f"{prefix}description of {c}: {da[c]} != {db[c]}")
        ka, kb = _no_links(a.getkeywords()), _no_links(b.getkeywords())
        if not _eq(ka, kb):
            diffs.append(f"{prefix}table keywords: {ka} != {kb}")
        if not _eq(_dminfo(a), _dminfo(b)):
            diffs.append(f"{prefix}dminfo: {_dminfo(a)} != {_dminfo(b)}")
        if a.info() != b.info():
            diffs.append(f"{prefix}info {a.info()} != {b.info()}")
        if values and a.nrows() == b.nrows():
            for c in sorted(set(a.colnames()) & set(b.colnames())):
                if c in skip_cols or not a.nrows():
                    continue
                if not _eq(_column_values(a, c), _column_values(b, c)):
                    diffs.append(f"{prefix}values of {c} differ")
    if with_subtables:
        sa, sb = subtables(pa), subtables(pb)
        if sa != sb:
            diffs.append(f"{prefix}subtables {sa} != {sb}")
        for s in sorted(set(sa) & set(sb)):
            diffs += table_diff(str(Path(pa) / s), str(Path(pb) / s), values=values,
                                with_subtables=False, prefix=f"{prefix}{s}/")
    return diffs
