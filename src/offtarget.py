"""Off-target site search and CFD scoring.

A guide with a superb on-target score that also cuts eleven other places in the
genome is worse than useless, so off-target analysis is not an optional extra --
it is half the problem. Two pieces:

**Search.** Find every site in a reference matching the guide within a mismatch
budget, on both strands, with a valid PAM. Done naively this is
``len(reference) x 20`` character comparisons; the seed-and-extend index here
skips the overwhelming majority of positions.

**Scoring.** Not all mismatches are equal. A mismatch in the PAM-proximal
"seed" region (roughly positions 12-20) is far more disruptive than one at the
5' end, and the identity of the substitution matters too. CFD (Cutting Frequency
Determination) captures this with a position-and-substitution-specific penalty
table.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .sequence import GUIDE_LENGTH, reverse_complement

# Position weights: relative tolerance of a mismatch at each protospacer
# position, 1-based from the 5' end. Derived from the shape reported by Doench
# et al. (2016) -- high values near the 5' end (mismatches tolerated, so the
# site still cuts) falling sharply through the seed region toward the PAM.
POSITION_WEIGHT: Tuple[float, ...] = (
    0.94, 0.91, 0.89, 0.88, 0.86, 0.83, 0.80, 0.76, 0.71, 0.65,
    0.58, 0.50, 0.42, 0.34, 0.26, 0.19, 0.13, 0.08, 0.04, 0.02,
)

# Multipliers on the position weight for specific substitutions. rG:dT and
# rU:dG wobble pairs are chemically near-tolerated, so they barely reduce
# cutting; purine-purine mismatches are strongly disruptive.
SUBSTITUTION_MULTIPLIER: Dict[Tuple[str, str], float] = {
    ("G", "T"): 1.35, ("T", "G"): 1.35,      # wobble-like, well tolerated
    ("A", "C"): 1.15, ("C", "A"): 1.15,
    ("G", "A"): 0.75, ("A", "G"): 0.75,      # purine-purine, disruptive
    ("C", "T"): 0.90, ("T", "C"): 0.90,
    ("A", "T"): 0.85, ("T", "A"): 0.85,
    ("C", "G"): 0.70, ("G", "C"): 0.70,
}

# PAM tolerance. NGG is canonical; NAG and NGA cut at reduced efficiency and
# are the usual source of unexpected off-target activity, so they are searched
# rather than ignored.
PAM_ACTIVITY: Dict[str, float] = {"GG": 1.0, "AG": 0.26, "GA": 0.07}


@dataclass
class OffTarget:
    """A site in the reference that the guide may also cut."""

    sequence: str
    pam: str
    position: int
    strand: str
    n_mismatches: int
    mismatch_positions: List[int]
    cfd_score: float

    def __str__(self) -> str:
        return (f"{self.sequence} {self.pam} ({self.strand}) @{self.position} "
                f"mm={self.n_mismatches} cfd={self.cfd_score:.4f}")


def cfd_score(guide: str, target: str, pam: str = "GG",
              custom_matrix: Dict[Tuple[int, str, str], float] | None = None
              ) -> float:
    """Cutting Frequency Determination score for a guide against a site.

    Returns a value in [0, 1]: 1.0 is a perfect match cutting at full
    efficiency, values near 0 mean the site is effectively not cut. Penalties
    are multiplicative across mismatches, which is the empirically observed
    behaviour -- two seed mismatches abolish cutting rather than merely halving
    it.

    The weights here reproduce the *shape* of the published CFD matrix. For
    clinical or publication work, pass the full experimentally measured table
    from Doench et al. 2016 as ``custom_matrix`` -- :func:`load_cfd_matrix`
    reads it from the published CSV. Any ``(position, guide_base,
    target_base)`` triple absent from the custom matrix falls back to the
    built-in approximation, so a partial table is safe to use.
    """
    guide = guide.upper()
    target = target.upper()
    if len(guide) != len(target):
        raise ValueError(
            f"guide ({len(guide)}) and target ({len(target)}) must be the "
            f"same length"
        )

    score = PAM_ACTIVITY.get(pam.upper()[-2:], 0.0)
    if score == 0.0:
        return 0.0

    for index, (g, t) in enumerate(zip(guide, target)):
        if g == t:
            continue
        weight = None
        if custom_matrix is not None:
            weight = custom_matrix.get((index + 1, g, t))
        if weight is None:
            weight = POSITION_WEIGHT[index] if index < len(POSITION_WEIGHT) else 0.5
            weight *= SUBSTITUTION_MULTIPLIER.get((g, t), 1.0)
        # Capped strictly below 1: a well-tolerated substitution at the 5' end
        # can push the product above 1, and clamping at exactly 1.0 would make
        # that mismatch entirely free -- two such mismatches would then score
        # identically to one, and the specificity ranking would be wrong.
        score *= min(0.99, weight)
    return float(score)


def load_cfd_matrix(path: str) -> Dict[Tuple[int, str, str], float]:
    """Load a published CFD table: ``position,guide_base,target_base,score``.

    ``position`` is 1-based from the 5' end of the protospacer, matching the
    convention :func:`cfd_score` and the rest of this module use. Pass the
    result as ``custom_matrix`` to :func:`cfd_score` or as ``cfd_matrix`` to
    :class:`OffTargetSearcher` to use it in place of the built-in table.
    """
    import csv

    matrix: Dict[Tuple[int, str, str], float] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["position"]), row["guide_base"].upper(),
                   row["target_base"].upper())
            matrix[key] = float(row["score"])
    return matrix


def hamming_distance(a: str, b: str, limit: int | None = None) -> int:
    """Mismatch count, short-circuiting once ``limit`` is exceeded.

    The early exit is what makes the search tractable: almost every candidate
    site fails within the first few comparisons, so there is no reason to
    finish counting.
    """
    total = 0
    for x, y in zip(a, b):
        if x != y:
            total += 1
            if limit is not None and total > limit:
                return total
    return total


class OffTargetSearcher:
    """Seed-and-extend index over a reference sequence.

    A guide with at most ``m`` mismatches over 20 positions must match exactly
    across at least one of ``m + 1`` disjoint blocks -- the pigeonhole
    principle. Indexing every block-sized k-mer in the reference means only
    positions sharing an exact block need full comparison, which in practice is
    a tiny fraction. This is the same idea BLAST and Bowtie are built on.
    """

    def __init__(self, reference: str, max_mismatches: int = 4,
                 pam_variants: Iterable[str] = ("GG", "AG", "GA"),
                 cfd_matrix: Dict[Tuple[int, str, str], float] | None = None
                 ) -> None:
        self.reference = reference.upper()
        self.max_mismatches = max_mismatches
        self.pam_variants = tuple(pam_variants)
        # Published CFD table from load_cfd_matrix(), or None to use the
        # built-in approximation for every mismatch.
        self.cfd_matrix = cfd_matrix
        # Block length from the pigeonhole bound; at least 4 so the index stays
        # selective enough to be worth consulting.
        self.seed_length = max(4, GUIDE_LENGTH // (max_mismatches + 1))
        self._index: Dict[str, List[int]] | None = None

    def build_index(self) -> None:
        index: Dict[str, List[int]] = defaultdict(list)
        reference = self.reference
        seed = self.seed_length
        for i in range(len(reference) - seed + 1):
            kmer = reference[i : i + seed]
            if "N" not in kmer:
                index[kmer].append(i)
        self._index = index

    def _candidate_starts(self, guide: str) -> set[int]:
        """Protospacer start positions worth a full comparison."""
        if self._index is None:
            self.build_index()
        assert self._index is not None

        seed = self.seed_length
        candidates: set[int] = set()
        # One seed per disjoint block; m+1 blocks guarantees one is intact.
        for block in range(self.max_mismatches + 1):
            offset = block * seed
            if offset + seed > GUIDE_LENGTH:
                break
            kmer = guide[offset : offset + seed]
            for hit in self._index.get(kmer, ()):
                start = hit - offset
                if 0 <= start <= len(self.reference) - GUIDE_LENGTH:
                    candidates.add(start)
        return candidates

    def search(self, guide: str, include_perfect: bool = False
               ) -> List[OffTarget]:
        """All sites within the mismatch budget, on both strands."""
        guide = guide.upper()
        if len(guide) != GUIDE_LENGTH:
            raise ValueError(
                f"guide must be {GUIDE_LENGTH} nt, got {len(guide)}"
            )

        results: List[OffTarget] = []
        for strand, reference in (("+", self.reference),
                                  ("-", reverse_complement(self.reference))):
            searcher = (self if strand == "+"
                        else OffTargetSearcher(reference, self.max_mismatches,
                                               self.pam_variants,
                                               self.cfd_matrix))
            if strand == "-":
                searcher.seed_length = self.seed_length
            results.extend(
                searcher._search_strand(guide, strand, include_perfect)
            )
        results.sort(key=lambda hit: -hit.cfd_score)
        return results

    def _search_strand(self, guide: str, strand: str, include_perfect: bool
                       ) -> List[OffTarget]:
        reference = self.reference
        results: List[OffTarget] = []

        for start in self._candidate_starts(guide):
            end = start + GUIDE_LENGTH
            pam = reference[end : end + 3]
            if len(pam) < 3 or pam[-2:] not in self.pam_variants:
                continue

            site = reference[start:end]
            if "N" in site:
                continue
            mismatches = hamming_distance(guide, site, self.max_mismatches)
            if mismatches > self.max_mismatches:
                continue
            if mismatches == 0 and not include_perfect:
                continue

            positions = [i + 1 for i, (g, t) in enumerate(zip(guide, site))
                         if g != t]
            results.append(OffTarget(
                sequence=site, pam=pam, position=start, strand=strand,
                n_mismatches=mismatches, mismatch_positions=positions,
                cfd_score=cfd_score(guide, site, pam,
                                    custom_matrix=self.cfd_matrix),
            ))
        return results


def aggregate_specificity(off_targets: Sequence[OffTarget]) -> float:
    """Guide specificity score in [0, 100], following the MIT/CRISPOR form.

    ``100 / (1 + sum of off-target CFD scores)``. A guide with no credible
    off-targets scores 100; one whose off-target scores sum to 1.0 scores 50.
    The sum rather than the maximum is deliberate: twenty sites at CFD 0.05 are
    a real problem even though no single one looks alarming.
    """
    total = sum(hit.cfd_score for hit in off_targets)
    return float(100.0 / (1.0 + total))


def summarise_off_targets(off_targets: Sequence[OffTarget]) -> Dict[str, float]:
    by_mismatch = defaultdict(int)
    for hit in off_targets:
        by_mismatch[hit.n_mismatches] += 1
    return {
        "n_off_targets": len(off_targets),
        "specificity": aggregate_specificity(off_targets),
        "max_cfd": max((h.cfd_score for h in off_targets), default=0.0),
        "sum_cfd": sum(h.cfd_score for h in off_targets),
        **{f"n_mismatch_{k}": v for k, v in sorted(by_mismatch.items())},
    }
