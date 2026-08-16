"""Guide efficiency datasets.

The synthetic generator encodes the determinants of Cas9 cutting efficiency that
are actually established in the literature, so a model trained on it learns
something real rather than fitting noise:

* **GC content** -- efficiency peaks around 40-60%. Too low and the guide binds
  weakly; too high and it forms secondary structure or binds too tightly to
  release.
* **Position-specific nucleotides** -- G immediately 5' of the PAM (position 20)
  is favourable; T there is unfavourable. Position 16 disfavours a G.
* **Poly-T** -- ``TTTT`` terminates Pol III transcription, so the sgRNA is never
  made. This is a hard zero, not a penalty, and it is modelled that way.
* **Melting temperature** -- an intermediate optimum, for the same reason as GC.
* **PAM-proximal context** -- the nucleotide immediately 3' of the PAM matters.

Real datasets (Doench 2016, Kim 2019, Wang 2014) are loadable via
:func:`load_csv` and are strongly preferred; the generator exists so the
pipeline is testable and runnable with no download.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .sequence import (
    BASES,
    GUIDE_LENGTH,
    gc_content,
    has_polyt,
    melting_temperature,
    one_hot_encode,
)


def random_guide(rng: np.random.Generator, gc_bias: float = 0.5) -> str:
    probabilities = [(1 - gc_bias) / 2, gc_bias / 2, gc_bias / 2, (1 - gc_bias) / 2]
    return "".join(rng.choice(list(BASES), size=GUIDE_LENGTH, p=probabilities))


# Position-specific dinucleotide effects, as (0-based position, dimer, weight).
# Deliberately NOT exposed in biophysical_features(): they are local sequence
# patterns, which is exactly what the convolutional branch exists to find. A
# target built only from GC, Tm and poly-T would be fully determined by the
# feature vector, so the feature baseline would have privileged access to the
# generating process and the network could never do better than tie.
SEQUENCE_MOTIFS: tuple[tuple[int, str, float], ...] = (
    (4, "GG", 0.10),
    (9, "TA", -0.12),
    (13, "CC", -0.09),
    (17, "GA", 0.11),
    (6, "AT", -0.07),
)
# A 4-mer that suppresses cutting wherever it appears -- translation-invariant,
# so it is the convolution rather than the positional term that should catch it.
INHIBITORY_MOTIF = "GCCA"
INHIBITORY_WEIGHT = -0.13


def motif_term(guide: str) -> float:
    """Sequence-pattern contribution invisible to the biophysical features."""
    total = 0.0
    for position, dimer, weight in SEQUENCE_MOTIFS:
        if guide[position : position + 2] == dimer:
            total += weight
    if INHIBITORY_MOTIF in guide:
        total += INHIBITORY_WEIGHT
    return total


def true_efficiency(guide: str, context: str, rng: np.random.Generator | None = None
                    ) -> float:
    """Ground-truth efficiency in [0, 1] under the modelled determinants."""
    guide = guide.upper()

    # Pol III terminator: the sgRNA is truncated and nothing is cut.
    if has_polyt(guide):
        base = 0.02
        return float(np.clip(base + (rng.normal(0, 0.01) if rng is not None else 0),
                             0, 1))

    gc = gc_content(guide)
    # Inverted-U in GC, centred at 0.5.
    gc_term = 1.0 - 4.0 * (gc - 0.5) ** 2

    # Intermediate Tm is best, for the same reason as GC: too low and the guide
    # binds weakly, too high and it does not release. Centred on the range the
    # nearest-neighbour model actually produces at 50 mM Na+ (roughly 20-75 C).
    tm = melting_temperature(guide)
    tm_term = 1.0 - ((tm - 50.0) / 25.0) ** 2

    position_term = 0.0
    if guide[19] == "G":
        position_term += 0.18          # PAM-proximal G is favourable
    elif guide[19] == "T":
        position_term -= 0.15
    if guide[15] == "G":
        position_term -= 0.10
    if guide[2] == "A":
        position_term += 0.05

    context_term = 0.0
    if len(context) > 0 and context[-1] in "AT":
        context_term += 0.06

    # Purine-rich seed regions cut better.
    seed = guide[12:20]
    purine_term = 0.12 * ((seed.count("A") + seed.count("G")) / len(seed) - 0.5)

    # Base chosen so the realised distribution spans the full range rather than
    # bunching near 1, which saturates precision@k and ndcg into uselessness.
    score = (0.22 + 0.30 * gc_term + 0.12 * tm_term + position_term
             + context_term + purine_term + motif_term(guide))
    if rng is not None:
        score += rng.normal(0, 0.07)
    return float(np.clip(score, 0.0, 1.0))


def make_guide_dataset(
    n_guides: int = 5000, seed: int = 0, context_length: int = 4
) -> Tuple[List[str], List[str], np.ndarray]:
    """Generate ``(guides, contexts, efficiencies)``.

    A fraction of guides are drawn with skewed GC so the dataset spans the whole
    efficiency range; sampling uniformly at random gives a narrow, mostly
    mid-range distribution on which every model looks equally mediocre.
    """
    rng = np.random.default_rng(seed)
    guides, contexts, efficiencies = [], [], []

    for _ in range(n_guides):
        gc_bias = float(rng.choice([0.25, 0.4, 0.5, 0.6, 0.75],
                                   p=[0.15, 0.25, 0.25, 0.2, 0.15]))
        guide = random_guide(rng, gc_bias)
        left = "".join(rng.choice(list(BASES), size=context_length))
        right = "".join(rng.choice(list(BASES), size=context_length))
        context = left + guide + "AGG" + right

        guides.append(guide)
        contexts.append(context)
        efficiencies.append(true_efficiency(guide, right, rng))

    return guides, contexts, np.array(efficiencies, dtype=np.float32)


def load_csv(path: str | Path, guide_column: str = "guide",
             efficiency_column: str = "efficiency",
             context_column: str | None = "context"
             ) -> Tuple[List[str], List[str], np.ndarray]:
    """Load a real guide-efficiency dataset.

    Rows whose guide is not exactly 20 valid nucleotides are skipped and
    counted, rather than silently corrupting the feature matrix.
    """
    import csv

    guides, contexts, efficiencies = [], [], []
    skipped = 0
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            guide = row[guide_column].strip().upper()
            if len(guide) != GUIDE_LENGTH or not set(guide) <= set(BASES):
                skipped += 1
                continue
            guides.append(guide)
            contexts.append(
                row.get(context_column, guide) if context_column else guide
            )
            efficiencies.append(float(row[efficiency_column]))
    if skipped:
        print(f"skipped {skipped} rows with malformed guides")
    return guides, contexts, np.array(efficiencies, dtype=np.float32)


def biophysical_features(guide: str, context: str = "") -> np.ndarray:
    """Hand-computed features fed alongside the convolutional branch.

    These are the quantities a CRISPR biologist would compute by hand. Giving
    them to the network directly means its capacity goes toward the sequence
    patterns they *cannot* express, instead of rediscovering GC content from
    one-hot input.
    """
    guide = guide.upper()
    gc = gc_content(guide)
    return np.array([
        gc,
        (gc - 0.5) ** 2,                          # distance from optimum
        melting_temperature(guide) / 100.0,
        float(has_polyt(guide)),                  # hard failure flag
        float(has_polyt(guide, 3)),
        gc_content(guide[12:20]),                 # seed-region GC
        gc_content(guide[:8]),
        float(guide[19] == "G"),
        float(guide[19] == "T"),
        float(guide[15] == "G"),
        (guide[12:20].count("A") + guide[12:20].count("G")) / 8.0,
        guide.count("GG") / max(1, len(guide) - 1),
        float(len(context) > 0 and context[-1] in "AT"),
        max((len(run) for run in _homopolymer_runs(guide)), default=0) / 10.0,
    ], dtype=np.float32)


def _homopolymer_runs(sequence: str) -> List[str]:
    runs, current = [], sequence[:1]
    for base in sequence[1:]:
        if base == current[-1]:
            current += base
        else:
            runs.append(current)
            current = base
    runs.append(current)
    return runs


BIOPHYSICAL_FEATURE_DIM = len(biophysical_features("A" * GUIDE_LENGTH))


def encode_dataset(guides: Sequence[str], contexts: Sequence[str]
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(one_hot (N, 4, L), features (N, F))``."""
    one_hot = np.stack([one_hot_encode(g, GUIDE_LENGTH) for g in guides])
    features = np.stack([
        biophysical_features(g, c) for g, c in zip(guides, contexts)
    ])
    return one_hot, features


def split_indices(n: int, val_fraction: float, test_fraction: float, seed: int
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test, n_val = int(n * test_fraction), int(n * val_fraction)
    return (order[n_test + n_val:], order[n_test : n_test + n_val],
            order[:n_test])
