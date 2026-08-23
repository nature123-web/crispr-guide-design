"""End-to-end guide design: enumerate, score, check specificity, rank.

    python -m src.design --checkpoint runs/base/best.pt --target-file gene.fa \
        --genome-file genome.fa --top 10

This is where the two halves meet. A guide is only useful if it is *both*
efficient at its intended site and specific against the rest of the genome, and
those two properties are uncorrelated -- ranking on either one alone reliably
selects guides that fail on the other.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

from .data import biophysical_features
from .metrics import format_report
from .model import build_model
from .offtarget import (
    OffTarget,
    OffTargetSearcher,
    aggregate_specificity,
    load_cfd_matrix,
)
from .sequence import (
    Guide,
    find_guides,
    gc_content,
    has_polyt,
    melting_temperature,
    one_hot_encode,
)


@dataclass
class ScoredGuide:
    guide: Guide
    efficiency: float
    specificity: float = 100.0
    off_targets: List[OffTarget] = field(default_factory=list)
    rejected: str | None = None

    @property
    def combined_score(self) -> float:
        """Efficiency and specificity combined into one ranking number.

        The product, not the mean. An average lets a guide with 0.95 efficiency
        and 10/100 specificity outrank a balanced one, which is exactly the
        guide you must not order -- it will cut efficiently in a dozen places.
        Multiplying makes either factor being poor fatal.
        """
        if self.rejected:
            return 0.0
        return self.efficiency * (self.specificity / 100.0)


def load_model(checkpoint: str | Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


@torch.no_grad()
def score_efficiency(model, guides: Sequence[Guide], device: torch.device,
                     batch_size: int = 256) -> np.ndarray:
    if not guides:
        return np.array([], dtype=np.float32)
    one_hot = np.stack([one_hot_encode(g.sequence, 20) for g in guides])
    features = np.stack([
        biophysical_features(g.sequence, g.context) for g in guides
    ])

    scores = []
    for start in range(0, len(guides), batch_size):
        x = torch.from_numpy(one_hot[start : start + batch_size]).float().to(device)
        f = torch.from_numpy(features[start : start + batch_size]).float().to(device)
        scores.append(model.predict_efficiency(x, f).cpu().numpy())
    return np.concatenate(scores)


def hard_filter(guide: Guide, min_gc: float = 0.25, max_gc: float = 0.80
                ) -> str | None:
    """Reasons to discard a guide outright, before any model is consulted.

    These are not soft preferences. A poly-T guide is never transcribed, and a
    guide at 90% GC will not form a usable duplex -- no efficiency score should
    be allowed to rescue either, so they are removed rather than down-weighted.
    """
    if has_polyt(guide.sequence):
        return "poly-T terminator (TTTT) -- sgRNA is truncated by Pol III"
    gc = gc_content(guide.sequence)
    if gc < min_gc:
        return f"GC too low ({gc:.0%})"
    if gc > max_gc:
        return f"GC too high ({gc:.0%})"
    if "N" in guide.sequence:
        return "ambiguous base in protospacer"
    return None


def design_guides(
    target: str,
    model,
    device: torch.device,
    genome: str | None = None,
    max_mismatches: int = 4,
    top_n: int = 10,
    keep_rejected: bool = False,
    min_gc: float = 0.25,
    max_gc: float = 0.80,
    cfd_matrix: dict | None = None,
) -> List[ScoredGuide]:
    """Full pipeline for one target sequence."""
    candidates = find_guides(target)
    scored: List[ScoredGuide] = []

    keep: List[Guide] = []
    for guide in candidates:
        reason = hard_filter(guide, min_gc, max_gc)
        if reason:
            if keep_rejected:
                scored.append(ScoredGuide(guide, 0.0, rejected=reason))
        else:
            keep.append(guide)

    efficiencies = score_efficiency(model, keep, device)
    passing = [ScoredGuide(g, float(e)) for g, e in zip(keep, efficiencies)]

    if genome:
        # Off-target search is the expensive step, so it runs only on the guides
        # that could plausibly be chosen -- scoring all of them would waste most
        # of the work on guides that will never be ordered.
        passing.sort(key=lambda s: -s.efficiency)
        budget = min(len(passing), max(top_n * 3, 20))
        searcher = OffTargetSearcher(genome, max_mismatches, cfd_matrix=cfd_matrix)
        searcher.build_index()
        for candidate in passing[:budget]:
            hits = searcher.search(candidate.guide.sequence)
            candidate.off_targets = hits
            candidate.specificity = aggregate_specificity(hits)

    scored.extend(passing)
    scored.sort(key=lambda s: -s.combined_score)
    return scored


def read_fasta(path: str | Path) -> str:
    lines = Path(path).read_text().splitlines()
    return "".join(l.strip() for l in lines
                   if l.strip() and not l.startswith(">")).upper()


def format_guides(scored: Sequence[ScoredGuide], top_n: int = 10,
                  show_off_targets: int = 3) -> str:
    lines = [
        f"{'#':<4}{'protospacer':<24}{'PAM':<6}{'str':<5}{'cut':<8}"
        f"{'eff':>7}{'spec':>8}{'score':>8}{'GC':>7}{'Tm':>7}"
    ]
    for rank, entry in enumerate(scored[:top_n], start=1):
        guide = entry.guide
        lines.append(
            f"{rank:<4}{guide.sequence:<24}{guide.pam:<6}{guide.strand:<5}"
            f"{guide.cut_site:<8}{entry.efficiency:>7.3f}"
            f"{entry.specificity:>8.1f}{entry.combined_score:>8.3f}"
            f"{gc_content(guide.sequence):>7.0%}"
            f"{melting_temperature(guide.sequence):>7.1f}"
        )
        if show_off_targets and entry.off_targets:
            lines.append(f"      off-targets: {len(entry.off_targets)} sites")
            for hit in entry.off_targets[:show_off_targets]:
                lines.append(
                    f"        {hit.sequence} {hit.pam} ({hit.strand}) "
                    f"@{hit.position}  {hit.n_mismatches}mm  "
                    f"CFD={hit.cfd_score:.4f}  "
                    f"positions={hit.mismatch_positions}"
                )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target", default=None, help="Target DNA sequence.")
    parser.add_argument("--target-file", default=None, help="FASTA of target.")
    parser.add_argument("--genome", default=None,
                        help="Reference sequence for off-target search.")
    parser.add_argument("--genome-file", default=None)
    parser.add_argument("--max-mismatches", type=int, default=4)
    parser.add_argument("--cfd-matrix", default=None,
                        help="CSV of position,guide_base,target_base,score "
                             "(Doench et al. 2016) to replace the built-in "
                             "CFD approximation.")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--show-rejected", action="store_true")
    parser.add_argument("--out", default=None, help="Write a TSV here.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.target and not args.target_file:
        parser.error("provide --target or --target-file")

    device = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, device)

    target = args.target.upper() if args.target else read_fasta(args.target_file)
    genome = None
    if args.genome:
        genome = args.genome.upper()
    elif args.genome_file:
        genome = read_fasta(args.genome_file)

    print(f"target: {len(target)} bp")
    if genome:
        print(f"reference for off-target search: {len(genome)} bp "
              f"(<= {args.max_mismatches} mismatches)")

    cfd_matrix = load_cfd_matrix(args.cfd_matrix) if args.cfd_matrix else None
    scored = design_guides(
        target, model, device, genome, args.max_mismatches, args.top,
        keep_rejected=args.show_rejected, cfd_matrix=cfd_matrix,
    )
    usable = [s for s in scored if not s.rejected]
    rejected = [s for s in scored if s.rejected]

    print(f"\n{len(usable) + len(rejected)} PAM sites found, "
          f"{len(usable)} pass hard filters")
    if rejected:
        print(f"{len(rejected)} rejected:")
        for entry in rejected[:5]:
            print(f"  {entry.guide.sequence}  {entry.rejected}")

    print()
    print(format_guides(usable, args.top))

    if args.out:
        header = ("rank\tprotospacer\tpam\tstrand\tcut_site\tefficiency\t"
                  "specificity\tcombined\tn_off_targets")
        rows = [
            f"{i}\t{s.guide.sequence}\t{s.guide.pam}\t{s.guide.strand}\t"
            f"{s.guide.cut_site}\t{s.efficiency:.4f}\t{s.specificity:.2f}\t"
            f"{s.combined_score:.4f}\t{len(s.off_targets)}"
            for i, s in enumerate(usable[: args.top], start=1)
        ]
        Path(args.out).write_text(header + "\n" + "\n".join(rows) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
