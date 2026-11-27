#!/usr/bin/env python3
"""Deterministic left-most global alignment of a structure sequence onto a reference.

Why this matters: when a chain sequence is a subsequence of the reference with repeated
residues -- reference "…AAA…" against chain "…A…" -- several alignments score identically
and Biopython returns all of them. Picking a different one per contact would scatter the
same physical residue across different reference columns, and the 0/1 label strings of a
cluster would no longer be comparable.

Rule adopted: on a score tie, always take the alignment whose reference positions are
lexicographically smallest, i.e. bind to the earliest reference residue.

Biopython's `aligner.align(ref, chain)[0]` empirically already returns that alignment;
`verify_leftmost` checks it against the other optimal alignments so the assumption is
tested rather than trusted.
"""
from __future__ import annotations

import numpy as np
from Bio import Align

_ALIGNER = Align.PairwiseAligner()
_ALIGNER.mode = "global"


def _positions(alignment) -> tuple[np.ndarray, np.ndarray]:
    ref_blocks, chain_blocks = alignment.aligned
    rp = np.concatenate([np.arange(s, e) for s, e in ref_blocks.tolist()]) \
        if len(ref_blocks) else np.zeros(0, dtype=np.int64)
    cp = np.concatenate([np.arange(s, e) for s, e in chain_blocks.tolist()]) \
        if len(chain_blocks) else np.zeros(0, dtype=np.int64)
    n = min(len(rp), len(cp))
    return rp[:n].astype(np.int64), cp[:n].astype(np.int64)


def leftmost_align(chain_seq: str, ref_seq: str) -> tuple[np.ndarray, np.ndarray]:
    """(ref_positions, chain_positions), paired 1:1, left-most on ties."""
    if not chain_seq or not ref_seq:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return _positions(_ALIGNER.align(ref_seq, chain_seq)[0])


def verify_leftmost(chain_seq: str, ref_seq: str, max_alt: int = 200) -> bool:
    """Is alignment[0] really the left-most of the optimal alignments?"""
    alns = _ALIGNER.align(ref_seq, chain_seq)
    best = _positions(alns[0])[0]
    try:
        n = len(alns)
    except OverflowError:
        n = max_alt
    for k in range(1, min(n, max_alt)):
        rp = _positions(alns[k])[0]
        m = min(len(best), len(rp))
        for i in range(m):
            if rp[i] != best[i]:
                if rp[i] < best[i]:
                    return False        # found a strictly more left-most alignment
                break
    return True


if __name__ == "__main__":
    import random, sys
    cases = [("AA", "A"), ("AAA", "A"), ("MAAK", "MAK"), ("ACCCG", "ACG"),
             ("AAAA", "AA"), ("PEPTIDEAAAK", "PEPTIDEAK"), ("QQQWQQQ", "QWQ")]
    print("构造用例:")
    for ref, ch in cases:
        rp, cp = leftmost_align(ch, ref)
        ok = verify_leftmost(ch, ref)
        print(f"  ref={ref:<12} chain={ch:<10} ref_pos={rp.tolist()}  leftmost={ok}")

    print("\n随机用例 (含重复残基):")
    random.seed(0)
    bad = 0
    for _ in range(300):
        ref = "".join(random.choice("ACDEFGA") for _ in range(random.randint(20, 60)))
        keep = sorted(random.sample(range(len(ref)), k=int(len(ref) * 0.8)))
        ch = "".join(ref[i] for i in keep)
        if not verify_leftmost(ch, ref):
            bad += 1
    print(f"  300 个随机用例中, alignment[0] 不是最左的: {bad}")
