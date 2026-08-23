"""Tests for sequence handling, off-target search, CFD, the model and metrics."""

import numpy as np
import pytest
import torch

from src.data import (
    BIOPHYSICAL_FEATURE_DIM,
    biophysical_features,
    encode_dataset,
    make_guide_dataset,
    true_efficiency,
)
from src.design import ScoredGuide, hard_filter
from src.metrics import (
    evaluate,
    ndcg_at_k,
    precision_at_k,
    spearman,
    top_guide_efficiency,
)
from src.model import GuideEfficiencyNet
from src.offtarget import (
    OffTargetSearcher,
    aggregate_specificity,
    cfd_score,
    hamming_distance,
    load_cfd_matrix,
    summarise_off_targets,
)
from src.sequence import (
    GUIDE_LENGTH,
    Guide,
    find_guides,
    find_pam_sites,
    gc_content,
    has_polyt,
    melting_temperature,
    one_hot_encode,
    reverse_complement,
)


# --------------------------------------------------------------------------- #
# Sequence handling
# --------------------------------------------------------------------------- #

def test_reverse_complement_is_an_involution():
    seq = "ACGTTGCAAGGCTNNAC"
    assert reverse_complement(reverse_complement(seq)) == seq


def test_reverse_complement_known_value():
    assert reverse_complement("AAAACGT") == "ACGTTTT"


def test_gc_content():
    assert gc_content("GGCC") == 1.0
    assert gc_content("ATAT") == 0.0
    assert gc_content("ACGT") == 0.5


def test_polyt_detection():
    """TTTT terminates Pol III, so the sgRNA is never made."""
    assert has_polyt("ACGTTTTACG")
    assert not has_polyt("ACGTTTACG")
    assert has_polyt("ACGTTTACG", run_length=3)


def test_melting_temperature_ordering():
    """GC-rich duplexes are more stable than AT-rich ones of equal length."""
    gc_rich = melting_temperature("GCGCGCGCGCGCGCGCGCGC")
    at_rich = melting_temperature("ATATATATATATATATATAT")
    assert gc_rich > at_rich


def test_melting_temperature_is_physically_plausible():
    """Guards the salt and concentration constants.

    A 20-mer at 50 mM Na+ melts somewhere in the 35-90 C range. Getting the
    sign of the salt correction or the R*ln(C/4) term wrong shifts every value
    by tens of degrees while preserving the ordering, so a purely comparative
    test would not catch it -- and every Tm-derived feature would be garbage.
    """
    for guide in ("ATATATATATATATATATAT", "ACGATCGATCGATCGATCGA",
                  "GACGCATAAAGATGAGACGC", "GCGCGCGCGCGCGCGCGCGC"):
        tm = melting_temperature(guide)
        assert 15.0 < tm < 90.0, f"{guide}: Tm {tm:.1f} C is not physical"


def test_helix_initiation_terms_are_applied():
    """Dropping initiation inflates Tm by 10-20 C without changing the order."""
    guide = "GACGCATAAAGATGAGACGC"
    assert melting_temperature(guide) < 60.0


def test_melting_temperature_responds_to_salt():
    guide = "GACGCATAAAGATGAGACGC"
    low_salt = melting_temperature(guide, sodium_molar=0.01)
    high_salt = melting_temperature(guide, sodium_molar=1.0)
    assert high_salt > low_salt


def test_melting_temperature_is_stacking_dependent():
    """Same composition, different order -- Tm must differ.

    This is why the nearest-neighbour model is used rather than the Wallace
    rule, which would give these two identical values.
    """
    a = melting_temperature("GCGCGCGCGCGCGCGCGCGC")
    b = melting_temperature("GGGGGGGGGGCCCCCCCCCC")
    assert abs(a - b) > 0.5


def test_find_pam_sites_finds_overlapping_matches():
    """AGGG contains two NGG sites; a non-overlapping scan reports one."""
    sites = list(find_pam_sites("AGGG"))
    assert len(sites) == 2


def test_find_pam_sites_respects_the_n():
    sites = [pam for _, pam in find_pam_sites("TGGAGGCGG")]
    assert all(pam[1:] == "GG" for pam in sites)


def test_find_guides_returns_20nt_protospacers():
    rng = np.random.default_rng(0)
    sequence = "".join(rng.choice(list("ACGT"), size=300))
    for guide in find_guides(sequence):
        assert len(guide.sequence) == GUIDE_LENGTH
        assert guide.pam[1:] == "GG"


def test_find_guides_searches_both_strands():
    """A reverse-strand target appears as CCN forward; missing it loses half."""
    # No forward NGG, but the reverse complement has one.
    sequence = "A" * 25 + "CCA" + "T" * 25
    forward_only = find_guides(sequence, both_strands=False)
    both = find_guides(sequence, both_strands=True)
    assert len(both) > len(forward_only)
    assert any(g.strand == "-" for g in both)


def test_guide_protospacer_is_immediately_5_prime_of_pam():
    sequence = "ACGT" * 10 + "AGG"
    guides = [g for g in find_guides(sequence, both_strands=False)
              if g.strand == "+"]
    assert guides
    for guide in guides:
        end = guide.start + GUIDE_LENGTH
        assert sequence[guide.start:end] == guide.sequence
        assert sequence[end : end + 3] == guide.pam


def test_cut_site_is_three_bases_upstream_of_pam():
    guide = Guide(sequence="A" * 20, pam="AGG", start=100, strand="+")
    assert guide.cut_site == 100 + 20 - 3


def test_cut_site_on_reverse_strand():
    guide = Guide(sequence="A" * 20, pam="AGG", start=100, strand="-")
    assert guide.cut_site == 103


def test_guides_near_the_start_are_not_emitted_truncated():
    """A PAM at position 5 has no room for a 20 nt protospacer."""
    for guide in find_guides("AGG" + "ACGT" * 20):
        assert guide.start >= 0
        assert len(guide.sequence) == GUIDE_LENGTH


def test_one_hot_shape_and_ambiguous_base():
    encoded = one_hot_encode("ACGTN")
    assert encoded.shape == (4, 5)
    assert np.allclose(encoded[:, 4], 0.25)
    assert np.allclose(encoded.sum(axis=0), 1.0)


def test_one_hot_pads_to_requested_length():
    assert one_hot_encode("ACGT", length=10).shape == (4, 10)


# --------------------------------------------------------------------------- #
# CFD scoring
# --------------------------------------------------------------------------- #

GUIDE = "GACGCATAAAGATGAGACGC"


def mutate(sequence: str, *positions: int) -> str:
    """Substitute a genuinely different base at each 0-based position.

    Writing a literal base risks picking the one already there, which silently
    produces fewer mismatches than the test intends -- exactly the mistake this
    helper exists to prevent.
    """
    bases = list(sequence)
    for position in positions:
        current = bases[position]
        bases[position] = next(b for b in "ACGT" if b != current)
    return "".join(bases)


def test_mutate_helper_actually_changes_bases():
    assert mutate(GUIDE, 0) != GUIDE
    assert sum(a != b for a, b in zip(GUIDE, mutate(GUIDE, 0, 5, 9))) == 3


def test_perfect_match_scores_one():
    assert cfd_score(GUIDE, GUIDE, "GG") == pytest.approx(1.0)


def test_cfd_falls_with_more_mismatches():
    one = mutate(GUIDE, 0)
    two = mutate(GUIDE, 0, 1)
    three = mutate(GUIDE, 0, 1, 2)
    assert (cfd_score(GUIDE, GUIDE) > cfd_score(GUIDE, one)
            > cfd_score(GUIDE, two) > cfd_score(GUIDE, three))


def test_seed_mismatches_hurt_far_more_than_distal_ones():
    """The central biological fact CFD encodes.

    A mismatch adjacent to the PAM nearly abolishes cutting; the same
    substitution at the 5' end is largely tolerated.
    """
    guide = "GACGCATAAAGATGAGACGC"
    distal = "A" + guide[1:]                       # position 1
    seed = guide[:19] + ("A" if guide[19] != "A" else "T")   # position 20
    assert cfd_score(guide, distal) > 5 * cfd_score(guide, seed)


def test_non_canonical_pam_reduces_the_score():
    guide = "GACGCATAAAGATGAGACGC"
    assert cfd_score(guide, guide, "GG") > cfd_score(guide, guide, "AG")
    assert cfd_score(guide, guide, "AG") > cfd_score(guide, guide, "GA")


def test_invalid_pam_scores_zero():
    guide = "GACGCATAAAGATGAGACGC"
    assert cfd_score(guide, guide, "TT") == 0.0


def test_cfd_is_bounded():
    rng = np.random.default_rng(0)
    guide = "".join(rng.choice(list("ACGT"), size=20))
    for _ in range(50):
        target = "".join(rng.choice(list("ACGT"), size=20))
        score = cfd_score(guide, target)
        assert 0.0 <= score <= 1.0


def test_cfd_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        cfd_score("ACGT", "ACG")


def test_load_cfd_matrix_reads_the_published_csv_format(tmp_path):
    path = tmp_path / "cfd.csv"
    path.write_text(
        "position,guide_base,target_base,score\n"
        "1,G,A,0.5\n"
        "20,C,T,0.02\n"
    )
    matrix = load_cfd_matrix(str(path))
    assert matrix[(1, "G", "A")] == pytest.approx(0.5)
    assert matrix[(20, "C", "T")] == pytest.approx(0.02)


def test_custom_cfd_matrix_overrides_the_builtin_table():
    """The README promises load_cfd_matrix's table replaces the built-in
    approximation; this checks it is actually consulted, not just parsed.
    """
    guide = GUIDE
    target = mutate(guide, 0)                     # position 1, G -> A
    builtin = cfd_score(guide, target, "AGG")

    custom = {(1, "G", "A"): 0.5}
    overridden = cfd_score(guide, target, "AGG", custom_matrix=custom)

    assert overridden == pytest.approx(0.5)        # PAM activity 1.0 * 0.5
    assert overridden != pytest.approx(builtin)


def test_custom_cfd_matrix_falls_back_for_missing_entries():
    """A partial table should not silently zero out unlisted mismatches."""
    guide = GUIDE
    target = mutate(guide, 5)                      # not in the custom matrix
    custom = {(1, "G", "A"): 0.5}                  # unrelated entry only

    builtin = cfd_score(guide, target, "AGG")
    with_partial_matrix = cfd_score(guide, target, "AGG", custom_matrix=custom)
    assert with_partial_matrix == pytest.approx(builtin)


def test_hamming_distance_short_circuits():
    assert hamming_distance("AAAA", "AAAA") == 0
    assert hamming_distance("AAAA", "TTTT") == 4
    # With a limit of 1 it stops early and reports exceeding, not the true count.
    assert hamming_distance("AAAA", "TTTT", limit=1) == 2


# --------------------------------------------------------------------------- #
# Off-target search
# --------------------------------------------------------------------------- #

def build_reference(rng, length=4000):
    return "".join(rng.choice(list("ACGT"), size=length))


def test_search_finds_a_planted_exact_site():
    rng = np.random.default_rng(0)
    guide = "GACGCATAAAGATGAGACGC"
    reference = build_reference(rng, 2000)
    reference = reference[:500] + guide + "TGG" + reference[523:]

    searcher = OffTargetSearcher(reference, max_mismatches=3)
    hits = searcher.search(guide, include_perfect=True)
    assert any(h.position == 500 and h.n_mismatches == 0 for h in hits)


def test_search_finds_a_planted_mismatched_site():
    rng = np.random.default_rng(1)
    guide = GUIDE
    variant = mutate(guide, 0, 8)                    # exactly 2 mismatches
    assert sum(a != b for a, b in zip(guide, variant)) == 2

    reference = build_reference(rng, 2000)
    reference = reference[:800] + variant + "AGG" + reference[823:]

    hits = OffTargetSearcher(reference, max_mismatches=3).search(guide)
    found = [h for h in hits if h.position == 800 and h.strand == "+"]
    assert found and found[0].n_mismatches == 2
    assert found[0].mismatch_positions == [1, 9]     # 1-based


def test_search_uses_a_custom_cfd_matrix_when_given():
    """cfd_matrix passed to the searcher must reach every scored off-target,
    including the reverse-strand path, which builds its own sub-searcher.
    """
    rng = np.random.default_rng(1)
    guide = GUIDE
    variant = mutate(guide, 0)                     # position 1, G -> A
    reference = build_reference(rng, 2000)
    reference = reference[:800] + variant + "AGG" + reference[823:]

    custom = {(1, "G", "A"): 0.5}
    hits = OffTargetSearcher(reference, max_mismatches=3,
                             cfd_matrix=custom).search(guide)
    found = [h for h in hits if h.position == 800 and h.strand == "+"]
    assert found and found[0].cfd_score == pytest.approx(0.5)

    without_matrix = OffTargetSearcher(reference, max_mismatches=3).search(guide)
    default_hit = next(h for h in without_matrix
                       if h.position == 800 and h.strand == "+")
    assert default_hit.cfd_score != pytest.approx(0.5)


def test_search_respects_the_mismatch_budget():
    rng = np.random.default_rng(2)
    guide = "GACGCATAAAGATGAGACGC"
    reference = build_reference(rng, 3000)
    for hit in OffTargetSearcher(reference, max_mismatches=2).search(guide):
        assert hit.n_mismatches <= 2


def test_search_requires_a_valid_pam():
    """A perfect protospacer match without a PAM is not a cut site."""
    rng = np.random.default_rng(3)
    guide = "GACGCATAAAGATGAGACGC"
    reference = build_reference(rng, 1000)
    reference = reference[:400] + guide + "TTT" + reference[423:]

    hits = OffTargetSearcher(reference, max_mismatches=3).search(
        guide, include_perfect=True
    )
    assert not any(h.position == 400 and h.strand == "+" for h in hits)


def test_search_finds_reverse_strand_sites():
    rng = np.random.default_rng(4)
    guide = "GACGCATAAAGATGAGACGC"
    site = reverse_complement(guide + "AGG")
    reference = build_reference(rng, 2000)
    reference = reference[:600] + site + reference[623:]

    hits = OffTargetSearcher(reference, max_mismatches=2).search(
        guide, include_perfect=True
    )
    assert any(h.strand == "-" and h.n_mismatches == 0 for h in hits)


def test_seed_index_matches_brute_force():
    """The pigeonhole index must not miss anything a full scan would find.

    This is the test that matters for the whole module: an index that silently
    drops sites produces a guide that looks specific and is not.
    """
    rng = np.random.default_rng(5)
    guide = "GACGCATAAAGATGAGACGC"
    reference = build_reference(rng, 6000)
    # Plant several variants at known offsets.
    for offset, n_mm in ((500, 1), (1500, 2), (2500, 3), (3500, 4)):
        variant = list(guide)
        for k in range(n_mm):
            position = (k * 5 + 1) % 20
            variant[position] = "ACGT"[("ACGT".index(variant[position]) + 1) % 4]
        reference = (reference[:offset] + "".join(variant) + "AGG"
                     + reference[offset + 23:])

    max_mm = 4
    indexed = OffTargetSearcher(reference, max_mismatches=max_mm)
    found = {(h.position, h.strand) for h in indexed.search(guide,
                                                           include_perfect=True)}

    # Brute force over the forward strand for comparison.
    brute = set()
    for start in range(len(reference) - 23):
        site = reference[start : start + 20]
        pam = reference[start + 20 : start + 23]
        if pam[-2:] not in ("GG", "AG", "GA"):
            continue
        if sum(a != b for a, b in zip(guide, site)) <= max_mm:
            brute.add((start, "+"))

    missed = brute - found
    assert not missed, f"index missed {len(missed)} sites found by brute force"


def test_index_is_faster_than_brute_force_on_a_large_reference():
    """Sanity check that the index is doing real work."""
    rng = np.random.default_rng(6)
    reference = build_reference(rng, 50_000)
    searcher = OffTargetSearcher(reference, max_mismatches=3)
    searcher.build_index()
    guide = "GACGCATAAAGATGAGACGC"
    candidates = searcher._candidate_starts(guide)
    # Only a small fraction of positions should survive seeding.
    assert len(candidates) < 0.05 * len(reference)


def test_specificity_is_100_with_no_off_targets():
    assert aggregate_specificity([]) == pytest.approx(100.0)


def test_specificity_falls_as_off_targets_accumulate():
    rng = np.random.default_rng(7)
    guide = "GACGCATAAAGATGAGACGC"
    reference = build_reference(rng, 3000)
    clean = aggregate_specificity(
        OffTargetSearcher(reference, 2).search(guide)
    )
    # Plant three near-identical sites.
    for offset in (300, 1200, 2100):
        variant = "A" + guide[1:]
        reference = (reference[:offset] + variant + "AGG"
                     + reference[offset + 23:])
    dirty = aggregate_specificity(
        OffTargetSearcher(reference, 2).search(guide)
    )
    assert dirty < clean


def test_many_weak_off_targets_matter():
    """Summing rather than maxing: twenty weak sites are a real problem."""
    from src.offtarget import OffTarget

    weak = [OffTarget("A" * 20, "AGG", i, "+", 3, [1, 2, 3], 0.05)
            for i in range(20)]
    single_strong = [OffTarget("A" * 20, "AGG", 0, "+", 1, [1], 0.5)]
    assert aggregate_specificity(weak) < aggregate_specificity(single_strong)


def test_summarise_reports_mismatch_breakdown():
    rng = np.random.default_rng(8)
    guide = "GACGCATAAAGATGAGACGC"
    reference = build_reference(rng, 3000)
    summary = summarise_off_targets(
        OffTargetSearcher(reference, 4).search(guide)
    )
    assert "n_off_targets" in summary and "specificity" in summary
    assert 0 <= summary["specificity"] <= 100


def test_search_rejects_wrong_length_guide():
    with pytest.raises(ValueError, match="must be 20 nt"):
        OffTargetSearcher("ACGT" * 100, 2).search("ACGT")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def test_dataset_shapes():
    guides, contexts, y = make_guide_dataset(200, seed=0)
    assert len(guides) == len(contexts) == len(y) == 200
    assert all(len(g) == GUIDE_LENGTH for g in guides)
    assert (y >= 0).all() and (y <= 1).all()


def test_polyt_guides_are_near_zero_efficiency():
    """The hard biological constraint must survive into the labels."""
    rng = np.random.default_rng(0)
    bad = true_efficiency("ACGTTTTACGACGACGACGA", "AAAA", rng)
    good = true_efficiency("ACGACGACGACGACGACGAG", "AAAA", rng)
    assert bad < 0.1
    assert good > bad


def test_efficiency_peaks_at_intermediate_gc():
    """Inverted-U, not monotone -- both extremes are bad."""
    low = true_efficiency("ATATATATATATATATATAT", "AAAA")
    mid = true_efficiency("ACGATCGATCGATCGATCGA", "AAAA")
    high = true_efficiency("GCGCGCGCGCGCGCGCGCGC", "AAAA")
    assert mid > low and mid > high


def test_dataset_spans_a_wide_efficiency_range():
    """Bunched-up labels saturate precision@k and ndcg into uselessness."""
    _, _, y = make_guide_dataset(1500, seed=0)
    assert y.max() - y.min() > 0.5
    assert y.std() > 0.1
    assert 0.3 < y.mean() < 0.7


def test_target_contains_signal_outside_the_feature_vector():
    """Otherwise the feature baseline has privileged access to the truth.

    Gradient boosting on every biophysical feature must leave real residual
    variance, because the position-specific dinucleotide motifs are not among
    them. Without this the comparison in train.py would be rigged.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    guides, contexts, y = make_guide_dataset(2000, seed=0)
    _, features = encode_dataset(guides, contexts)
    model = GradientBoostingRegressor(n_estimators=200, random_state=0)
    model.fit(features[:1400], y[:1400])
    predicted = model.predict(features[1400:])
    assert spearman(y[1400:], predicted) < 0.97


def test_motif_term_responds_to_planted_patterns():
    from src.data import INHIBITORY_MOTIF, motif_term

    neutral = "ACACACACACACACACACAC"
    assert motif_term(neutral.replace(INHIBITORY_MOTIF, "", 1)) == \
        motif_term(neutral)

    with_inhibitor = "AC" + INHIBITORY_MOTIF + "ACACACACACACAC"
    assert motif_term(with_inhibitor) < motif_term(neutral)


def test_motif_term_is_position_sensitive():
    """The same dimer at a different position must score differently.

    This is the property the convolutional branch is there to exploit, and the
    one that composition-style features cannot express: GC content and poly-T
    status are identical for both sequences below.
    """
    from src.data import SEQUENCE_MOTIFS, motif_term

    position, dimer, weight = SEQUENCE_MOTIFS[0]
    bases = list("ACACACACACACACACACAC")

    at_motif = list(bases)
    at_motif[position], at_motif[position + 1] = dimer[0], dimer[1]

    elsewhere = list(bases)
    other = next(p for p in range(0, 18)
                 if all(abs(p - m[0]) > 1 for m in SEQUENCE_MOTIFS))
    elsewhere[other], elsewhere[other + 1] = dimer[0], dimer[1]

    a, b = "".join(at_motif), "".join(elsewhere)
    assert gc_content(a) == gc_content(b)
    assert has_polyt(a) == has_polyt(b)
    assert motif_term(a) != motif_term(b)
    assert motif_term(a) == pytest.approx(weight)


def test_biophysical_feature_dimension_is_stable():
    features = biophysical_features("ACGTACGTACGTACGTACGT", "AAAA")
    assert features.shape == (BIOPHYSICAL_FEATURE_DIM,)
    assert np.isfinite(features).all()


def test_polyt_flag_is_in_the_features():
    with_polyt = biophysical_features("ACGTTTTACGACGACGACGA")
    without = biophysical_features("ACGACGACGACGACGACGAG")
    assert not np.allclose(with_polyt, without)


def test_encode_dataset_shapes():
    guides, contexts, _ = make_guide_dataset(50, seed=0)
    one_hot, features = encode_dataset(guides, contexts)
    assert one_hot.shape == (50, 4, GUIDE_LENGTH)
    assert features.shape == (50, BIOPHYSICAL_FEATURE_DIM)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def make_model(**kwargs):
    defaults = dict(n_filters=16, kernel_sizes=(3, 5), hidden_dim=32,
                    dropout=0.0)
    defaults.update(kwargs)
    return GuideEfficiencyNet(**defaults)


def test_forward_shape():
    model = make_model().eval()
    one_hot = torch.randn(4, 4, GUIDE_LENGTH)
    features = torch.randn(4, BIOPHYSICAL_FEATURE_DIM)
    assert model(one_hot, features).shape == (4,)


def test_predict_efficiency_is_in_unit_interval():
    model = make_model().eval()
    out = model.predict_efficiency(
        torch.randn(8, 4, GUIDE_LENGTH), torch.randn(8, BIOPHYSICAL_FEATURE_DIM)
    )
    assert (out >= 0).all() and (out <= 1).all()


def test_feature_branch_can_be_ablated():
    model = make_model(use_features=False).eval()
    assert model(torch.randn(4, 4, GUIDE_LENGTH)).shape == (4,)


def test_missing_features_raise_a_clear_error():
    model = make_model(use_features=True).eval()
    with pytest.raises(ValueError, match="use_features=True"):
        model(torch.randn(2, 4, GUIDE_LENGTH))


def test_positional_parameter_breaks_translation_invariance():
    """A pure convolution cannot tell position 1 from position 20."""
    torch.manual_seed(0)
    model = make_model(use_features=False, use_positional=True).eval()
    with torch.no_grad():
        model.position.copy_(torch.randn_like(model.position))
        base = torch.zeros(1, 4, GUIDE_LENGTH)
        start, end = base.clone(), base.clone()
        start[0, 2, 0] = 1.0        # G at position 1
        end[0, 2, GUIDE_LENGTH - 1] = 1.0    # G at position 20
        assert not torch.allclose(model(start), model(end), atol=1e-4)


def test_gradients_flow():
    model = make_model()
    one_hot = torch.randn(4, 4, GUIDE_LENGTH)
    features = torch.randn(4, BIOPHYSICAL_FEATURE_DIM)
    model(one_hot, features).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())


def test_model_can_learn_the_synthetic_signal():
    """End-to-end sanity: a short fit must beat chance ranking."""
    torch.manual_seed(0)
    guides, contexts, y = make_guide_dataset(600, seed=0)
    one_hot, features = encode_dataset(guides, contexts)

    model = make_model(n_filters=32, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.from_numpy(one_hot).float()
    f = torch.from_numpy(features).float()
    target = torch.from_numpy(y).float()

    model.train()
    for _ in range(120):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(x[:400], f[:400]), target[:400]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predicted = model.predict_efficiency(x[400:], f[400:]).numpy()
    assert spearman(y[400:], predicted) > 0.5


# --------------------------------------------------------------------------- #
# Design pipeline
# --------------------------------------------------------------------------- #

def test_hard_filter_rejects_polyt():
    guide = Guide("ACGTTTTACGACGACGACGA", "AGG", 0, "+")
    assert "poly-T" in hard_filter(guide)


def test_hard_filter_rejects_extreme_gc():
    assert "GC too high" in hard_filter(Guide("G" * 20, "AGG", 0, "+"))
    assert "GC too low" in hard_filter(Guide("A" * 20, "AGG", 0, "+"))


def test_hard_filter_accepts_a_reasonable_guide():
    assert hard_filter(Guide("ACGATCGATCGATCGATCGA", "AGG", 0, "+")) is None


def test_combined_score_is_multiplicative():
    """A high-efficiency, low-specificity guide must not outrank a balanced one.

    Averaging would let 0.95 efficiency at 10 specificity beat 0.6 at 90, which
    is exactly the guide that must not be ordered.
    """
    reckless = ScoredGuide(Guide("A" * 20, "AGG", 0, "+"), 0.95, specificity=10.0)
    balanced = ScoredGuide(Guide("C" * 20, "AGG", 0, "+"), 0.60, specificity=90.0)
    assert balanced.combined_score > reckless.combined_score


def test_rejected_guides_score_zero():
    entry = ScoredGuide(Guide("A" * 20, "AGG", 0, "+"), 0.99,
                        specificity=100.0, rejected="poly-T")
    assert entry.combined_score == 0.0


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def test_spearman_of_a_perfect_ranking():
    y = np.array([0.1, 0.4, 0.6, 0.9])
    assert spearman(y, y) == pytest.approx(1.0)


def test_spearman_is_invariant_to_monotone_rescaling():
    """The reason it is the headline metric: only order matters."""
    y = np.array([0.1, 0.4, 0.6, 0.9])
    assert spearman(y, y ** 3) == pytest.approx(1.0)
    assert spearman(y, np.log(y)) == pytest.approx(1.0)


def test_spearman_of_a_reversed_ranking():
    y = np.array([0.1, 0.4, 0.6, 0.9])
    assert spearman(y, -y) == pytest.approx(-1.0)


def test_precision_at_k_of_a_perfect_ranker():
    y = np.arange(100) / 100.0
    assert precision_at_k(y, y, k=5, top_fraction=0.2) == 1.0


def test_precision_at_k_of_a_random_ranker():
    rng = np.random.default_rng(0)
    y = rng.random(2000)
    scores = [precision_at_k(y, rng.random(2000), k=5, top_fraction=0.2)
              for _ in range(30)]
    assert abs(np.mean(scores) - 0.2) < 0.12


def test_ndcg_is_bounded_and_perfect_for_ideal_ranking():
    y = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    assert ndcg_at_k(y, y, k=5) == pytest.approx(1.0)
    assert 0.0 <= ndcg_at_k(y, -y, k=5) <= 1.0


def test_ndcg_is_graded_not_binary():
    """Ranking the best guide first beats ranking the second-best first."""
    y = np.array([1.0, 0.8, 0.2, 0.1])
    best_first = np.array([4.0, 3.0, 2.0, 1.0])
    second_first = np.array([3.0, 4.0, 2.0, 1.0])
    assert ndcg_at_k(y, best_first, 4) > ndcg_at_k(y, second_first, 4)


def test_top_guide_efficiency_beats_the_mean_for_a_good_model():
    y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    assert top_guide_efficiency(y, y, k=1) == 0.9
    assert top_guide_efficiency(y, y, k=1) > y.mean()


def test_evaluate_returns_the_full_suite():
    rng = np.random.default_rng(0)
    y = rng.random(200)
    predicted = np.clip(y + rng.normal(0, 0.1, 200), 0, 1)
    results = evaluate(y, predicted)
    for key in ("spearman", "pearson", "rmse", "precision@5", "ndcg@10",
                "top1_efficiency"):
        assert key in results
    assert results["spearman"] > 0.7


def test_metrics_handle_a_constant_prediction():
    y = np.random.default_rng(0).random(50)
    results = evaluate(y, np.full(50, 0.5))
    assert np.isnan(results["spearman"])
    assert np.isfinite(results["rmse"])
