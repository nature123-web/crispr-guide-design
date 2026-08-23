# CRISPR Guide RNA Design

Machine learning for CRISPR-Cas9 guide selection: a neural network that predicts
**on-target cutting efficiency**, coupled to a seed-and-extend **off-target
search** with CFD scoring. Both halves are implemented from scratch — the
sequence handling, the nearest-neighbour thermodynamics, the genome index, and
the scoring model.

A guide is only useful if it is *both* efficient at its intended site and
specific against the rest of the genome. Those two properties are uncorrelated,
so ranking on either alone reliably picks guides that fail on the other. This
repo does both and combines them.

```
target sequence
   │  enumerate every NGG PAM, both strands
   ▼
candidate protospacers ──► hard filters (poly-T, GC extremes)
   │
   ├──► efficiency model    CNN + biophysical features → cut rate
   └──► off-target search   seed-and-extend over the genome → CFD → specificity
   │
   ▼  combined ranking (efficiency × specificity)
ordered guide list
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train --config configs/base.yaml
python -m src.train --config configs/base.yaml --no-features    # ablation
python -m src.design --checkpoint runs/base/best.pt \
    --target-file gene.fa --genome-file genome.fa --top 10

pytest    # 76 tests
```

Real output from `src.design` on a 600 bp target against a 60 kb reference:

```
67 PAM sites found, 63 pass hard filters
4 rejected:
  GGAGGCGATTTTTCGCTAGC  poly-T terminator (TTTT) -- sgRNA is truncated by Pol III
  ...

#   protospacer             PAM   str  cut         eff    spec   score     GC     Tm
1   CTAATGCTGGGGCTAACGAG    CGG   -    187       0.971   100.0   0.971    55%   49.1
2   CGCTCTGCAATCGCTATGAG    TGG   +    45        0.950   100.0   0.950    55%   49.9
3   AATCCGAAGTAGATACTGGG    CGG   +    103       0.935   100.0   0.935    45%   44.8
```

## Off-target search

Finding every site in a genome matching a guide within a mismatch budget is the
expensive half of guide design. Done naively it is `len(genome) × 20` character
comparisons.

The index here uses the **pigeonhole principle**: a guide with at most *m*
mismatches over 20 positions must match exactly across at least one of *m+1*
disjoint blocks. Indexing every block-sized k-mer means only positions sharing an
exact block need full comparison — under 5% of the genome in practice, which a
test asserts. This is the idea BLAST and Bowtie are built on.

The test that matters is `test_seed_index_matches_brute_force`: the index is
checked against an exhaustive scan, because an index that silently drops sites
produces a guide that *looks* specific and is not.

## CFD scoring: not all mismatches are equal

A mismatch in the PAM-proximal seed region nearly abolishes cutting; the same
substitution at the 5' end is largely tolerated. That is the single most
important fact in off-target analysis, and it is what CFD encodes. Measured
output from three planted single-mismatch decoys:

```
CTACTGCTGGGGCTAACGAG AGG (+) mm=1 @position 4   cfd=0.9900   ← still cuts
CAAATGCTGGGGCTAACGAG AGG (+) mm=1 @position 2   cfd=0.7735
CTAATGCTGGGGCTAACGAA AGG (+) mm=1 @position 20  cfd=0.0150   ← abolished

specificity: 36.0/100  (was 100.0 with no off-targets)
```

Same guide, same number of mismatches, a 66-fold difference in predicted cutting.
A tool that counted mismatches instead of scoring them would call these three
sites equivalent.

Non-canonical PAMs (`NAG`, `NGA`) are searched too, at reduced weight. They cut
inefficiently but they are the usual source of *unexpected* off-target activity,
so ignoring them is how you get surprised.

The built-in table reproduces the *shape* of the published CFD matrix, not the
full experimentally measured one. `--cfd-matrix path/to/table.csv` on
`src.design` (or `load_cfd_matrix` + the `cfd_matrix=`/`custom_matrix=`
argument on `OffTargetSearcher`/`cfd_score` directly) substitutes the real
Doench et al. 2016 table; any `(position, guide_base, target_base)` triple it
doesn't cover falls back to the built-in approximation.

## Hard filters are hard

`TTTT` is a termination signal for RNA polymerase III, which transcribes the
U6-driven sgRNA cassette. A guide containing one yields a truncated,
non-functional sgRNA **regardless of how good its predicted efficiency is**. It
is removed outright rather than down-weighted — no score should be able to
rescue it. Roughly 8% of random 20-mers are affected.

## The combined score is a product, not an average

```python
combined = efficiency * (specificity / 100)
```

Averaging would let a guide with 0.95 efficiency and 10/100 specificity outrank a
balanced one — and that is precisely the guide you must not order, because it
will cut efficiently in a dozen places. Multiplying makes either factor being
poor fatal. There is a test for it.

## Model design

Two branches, because they capture different things:

- **Convolutional branch** over one-hot sequence, with parallel kernel widths
  (3/5/7) rather than a deep stack — a guide is only 20 nt, so depth exhausts
  the sequence quickly.
- **Biophysical branch** carrying GC content, nearest-neighbour Tm, poly-T flags,
  seed-region composition and position-20 identity.

The split matters: a convolution is *translation-invariant*, which is right for
finding a motif anywhere and wrong for position-specific effects. A G at
position 20 is favourable *because it is at position 20*. A learned positional
parameter added to the input lets the convolutions break that invariance where
needed, and a test confirms they can distinguish position 1 from position 20.

Run `--no-features` to ablate the biophysical branch and see what it is worth.

## Metrics: Spearman, not RMSE

Guide design is a **ranking** problem. A researcher orders the top three or four
guides for a target and never uses the rest, so what matters is whether the
ordering is right, not whether the predicted efficiency is numerically accurate.
Published CRISPR models are compared on Spearman for exactly this reason.

`precision@5` and `ndcg@10` map directly to the workflow — order five guides, how
many work? — and `top1_efficiency` answers the only question a bench scientist
actually asks.

From a real run:

```
GuideEfficiencyNet:          spearman 0.8927   ndcg@10 0.9478   top1_efficiency 1.0000
ridge_features:              spearman 0.8766
gradient_boosting_features:  spearman 0.8773

network vs best feature baseline Spearman: +0.0154 (network wins)

picking the single top-ranked guide gives a true efficiency of 1.000,
against 0.557 for a guide chosen at random
```

The feature baselines are reported on every run because on guide efficiency they
are genuinely competitive — most of the signal is GC, poly-T and a few position
effects, all of which are in the feature vector. The network has to earn its
place above them, and the run says either way.

## The synthetic data is built to be fair

The generator encodes established determinants of Cas9 efficiency: an
inverted-U in GC content, position-specific nucleotides (G at position 20 good,
T bad), poly-T as a hard zero, an intermediate Tm optimum, and seed-region purine
content.

Crucially it *also* includes **position-specific dinucleotide motifs and an
inhibitory 4-mer that are deliberately absent from the feature vector**. Without
them the target would be a smooth function of the very features the baseline
receives, giving it privileged access to the generating process — the network
could never do better than tie. A test asserts gradient boosting on all features
leaves real residual signal, keeping the comparison honest as the generator
evolves.

Real data is strongly preferred and loads directly:

```bash
python -m src.train --config configs/base.yaml --csv data/doench2016.csv
```

| Dataset | Content |
| --- | --- |
| [Doench et al. 2016](https://www.nature.com/articles/nbt.3437) | ~2,500 guides, the standard on-target benchmark |
| [Kim et al. 2019](https://www.science.org/doi/10.1126/sciadv.aax9249) | large-scale deep-learning training set |
| [GUIDE-seq](https://www.nature.com/articles/nbt.3117) | experimentally measured off-target sites |

## A note on the thermodynamics

Tm uses **SantaLucia unified nearest-neighbour parameters with helix initiation
terms**, not the Wallace rule. Two guides with identical base composition can
have materially different stability depending on stacking, and composition alone
cannot see that.

Getting the salt and concentration constants wrong inflates every Tm by tens of
degrees while *preserving the ordering* — so a purely comparative test passes
while every Tm-derived feature is silently miscalibrated. There is an explicit
test that values fall in a physically plausible band.

## Layout

```
src/
  sequence.py   PAM finding, guide extraction, cut sites, NN thermodynamics
  offtarget.py  seed-and-extend index, CFD scoring, specificity aggregation
  data.py       efficiency determinants, biophysical features, CSV loading
  model.py      two-branch CNN
  metrics.py    Spearman, precision@k, NDCG, top-guide efficiency
  train.py      training loop + feature baselines
  design.py     end-to-end: enumerate → filter → score → search → rank
tests/          pytest suite (76 tests)
```

## Scope

Single-guide SpCas9 design against a supplied reference. Not covered: base and
prime editing, Cas12a/Cpf1 (`TTTV` PAM, different cut geometry), chromatin
accessibility, whole-genome indexing at human scale (use a suffix array or an
FM-index for that), and repair-outcome prediction. The CFD weights reproduce the
*shape* of the published matrix; for clinical work substitute the full
experimental table from Doench et al. 2016 via `load_cfd_matrix`.

## License

MIT
