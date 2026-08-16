"""DNA sequence utilities and CRISPR guide extraction.

The conventions encoded here are SpCas9's and they are not arbitrary:

* The protospacer adjacent motif (**PAM**) is ``NGG``, sitting immediately 3' of
  the 20-nucleotide protospacer. Cas9 physically cannot cut without it.
* The **cut site** is 3 bp upstream of the PAM, between positions 17 and 18 of
  the protospacer. Everything about editing outcome -- which base gets edited,
  where an indel lands -- is measured relative to that point.
* Both strands must be searched. A target on the reverse strand appears as
  ``CCN`` in the forward sequence, and a tool that only scans forward silently
  discards half the available guides.

Positions are reported 1-based from the 5' end of the protospacer, as they are
in the CRISPR literature, so position 20 is the nucleotide adjacent to the PAM.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterator, List, Sequence

BASES = "ACGT"
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

GUIDE_LENGTH = 20
PAM_LENGTH = 3
# Distance from the 3' end of the protospacer back to the cut site.
CUT_OFFSET = 3


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def is_valid_dna(sequence: str, allow_n: bool = False) -> bool:
    allowed = set(BASES + ("N" if allow_n else ""))
    return len(sequence) > 0 and set(sequence.upper()) <= allowed


def gc_content(sequence: str) -> float:
    if not sequence:
        return 0.0
    sequence = sequence.upper()
    return (sequence.count("G") + sequence.count("C")) / len(sequence)


# SantaLucia (1998) unified nearest-neighbour parameters.
# dH in kcal/mol, dS in cal/mol/K. Preferred over the older Breslauer set,
# which substantially overestimates Tm for GC-rich duplexes.
_NN_ENTHALPY = {
    "AA": -7.9, "TT": -7.9, "AT": -7.2, "TA": -7.2,
    "CA": -8.5, "TG": -8.5, "GT": -8.4, "AC": -8.4,
    "CT": -7.8, "AG": -7.8, "GA": -8.2, "TC": -8.2,
    "CG": -10.6, "GC": -9.8, "GG": -8.0, "CC": -8.0,
}
_NN_ENTROPY = {
    "AA": -22.2, "TT": -22.2, "AT": -20.4, "TA": -21.3,
    "CA": -22.7, "TG": -22.7, "GT": -22.4, "AC": -22.4,
    "CT": -21.0, "AG": -21.0, "GA": -22.2, "TC": -22.2,
    "CG": -27.2, "GC": -24.4, "GG": -19.9, "CC": -19.9,
}
# Helix initiation, which depends on whether each end is a G:C or an A:T pair.
_INIT_GC = (0.1, -2.8)
_INIT_AT = (2.3, 4.1)


def melting_temperature(sequence: str, sodium_molar: float = 0.05,
                        oligo_molar: float = 0.5e-6) -> float:
    """Nearest-neighbour Tm in degrees Celsius (SantaLucia unified parameters).

    The Wallace rule (2*AT + 4*GC) is far too crude here: guide binding energy
    depends on stacking between adjacent bases, so ``GCGCGC`` and ``GGGCCC``
    have the same composition and materially different stability. Nearest
    neighbour captures that; composition alone does not.

    Helix **initiation** terms are included. Omitting them inflates Tm by
    roughly 10-20 C, and because the inflation is systematic the ordering of
    guides still looks correct -- so a purely comparative test will not catch
    it while every Tm-derived feature is silently miscalibrated.
    """
    sequence = sequence.upper()
    if len(sequence) < 2:
        return 0.0

    total_h, total_s = 0.0, 0.0
    counted = 0
    for i in range(len(sequence) - 1):
        pair = sequence[i : i + 2]
        if pair not in _NN_ENTHALPY:      # skip ambiguous bases
            continue
        total_h += _NN_ENTHALPY[pair]
        total_s += _NN_ENTROPY[pair]
        counted += 1

    if counted == 0:
        return 0.0

    for end in (sequence[0], sequence[-1]):
        init_h, init_s = _INIT_GC if end in "GC" else _INIT_AT
        total_h += init_h
        total_s += init_s

    # Tm = 1000*dH / (dS + R ln(C_T/4)) - 273.15, then corrected for salt.
    # For a non-self-complementary duplex the concentration term is ln(C_T/4).
    gas_constant = 1.987
    total_s += gas_constant * math.log(oligo_molar / 4.0)
    if abs(total_s) < 1e-9:
        return 0.0

    tm_kelvin = (1000 * total_h) / total_s
    return tm_kelvin - 273.15 + 16.6 * math.log10(sodium_molar)


def has_polyt(sequence: str, run_length: int = 4) -> bool:
    """True if the guide contains a run of T's that would kill transcription.

    ``TTTT`` is a termination signal for RNA polymerase III, which is what
    transcribes the U6-driven sgRNA cassette. A guide containing one produces a
    truncated, non-functional sgRNA regardless of how good its predicted
    on-target score is. This is a hard filter, not a soft penalty.
    """
    return "T" * run_length in sequence.upper()


@dataclass
class Guide:
    """A candidate protospacer with its genomic context."""

    sequence: str          # 20 nt protospacer, 5'->3' on the targeted strand
    pam: str               # 3 nt PAM
    start: int             # 0-based start of the protospacer in the input
    strand: str            # '+' or '-'
    context: str = ""      # extended window used by the model

    @property
    def cut_site(self) -> int:
        """0-based coordinate of the cut, in input-sequence coordinates.

        Reported on the forward strand regardless of which strand the guide
        targets, because that is the coordinate system an edit is validated in.
        """
        if self.strand == "+":
            return self.start + GUIDE_LENGTH - CUT_OFFSET
        return self.start + CUT_OFFSET

    @property
    def full_site(self) -> str:
        return self.sequence + self.pam

    def __str__(self) -> str:
        return f"{self.sequence} {self.pam} ({self.strand}) @{self.cut_site}"


def find_pam_sites(sequence: str, pam: str = "NGG") -> Iterator[tuple[int, str]]:
    """Yield ``(index, matched_pam)`` for every PAM on the forward strand.

    Uses a lookahead so overlapping PAMs are all found -- ``AGGG`` contains two
    valid ``NGG`` sites and a non-overlapping scan reports one.
    """
    pattern = "".join(
        "[ACGT]" if base == "N" else base for base in pam.upper()
    )
    for match in re.finditer(f"(?=({pattern}))", sequence.upper()):
        yield match.start(), match.group(1)


def find_guides(
    sequence: str,
    pam: str = "NGG",
    both_strands: bool = True,
    context_length: int = 4,
) -> List[Guide]:
    """Enumerate all candidate guides in a target sequence.

    ``context_length`` nucleotides either side are captured because flanking
    sequence measurably affects cutting efficiency -- the nucleotide immediately
    3' of the PAM in particular.
    """
    sequence = sequence.upper()
    guides: List[Guide] = []

    def scan(strand_sequence: str, strand: str) -> None:
        length = len(strand_sequence)
        for pam_index, matched_pam in find_pam_sites(strand_sequence, pam):
            start = pam_index - GUIDE_LENGTH
            if start < 0:
                continue
            protospacer = strand_sequence[start:pam_index]
            if not is_valid_dna(protospacer):
                continue

            left = max(0, start - context_length)
            right = min(length, pam_index + len(matched_pam) + context_length)
            context = strand_sequence[left:right]

            if strand == "+":
                original_start = start
            else:
                # Translate back to forward-strand coordinates.
                original_start = length - pam_index - len(matched_pam)

            guides.append(Guide(
                sequence=protospacer, pam=matched_pam,
                start=original_start, strand=strand, context=context,
            ))

    scan(sequence, "+")
    if both_strands:
        scan(reverse_complement(sequence), "-")
    return guides


def one_hot_encode(sequence: str, length: int | None = None) -> "np.ndarray":
    """Encode DNA as a (4, L) array; ambiguous bases become uniform 0.25."""
    import numpy as np

    sequence = sequence.upper()
    if length is not None:
        sequence = sequence[:length].ljust(length, "N")
    array = np.zeros((4, len(sequence)), dtype=np.float32)
    for i, base in enumerate(sequence):
        index = BASES.find(base)
        if index < 0:
            array[:, i] = 0.25
        else:
            array[index, i] = 1.0
    return array


def position_nucleotide_features(sequence: str) -> "np.ndarray":
    """Flattened one-hot, i.e. position-specific nucleotide identity.

    Kept separate from the convolutional input because position matters
    *absolutely* here, not translationally. A G at position 20 (PAM-proximal)
    is favourable and a T there is not -- a convolution, which is deliberately
    translation-invariant, cannot represent that on its own.
    """
    return one_hot_encode(sequence, GUIDE_LENGTH).T.flatten()
