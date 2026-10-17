# mc-greenium-validation

Monte Carlo design validation for a triple-difference study of greenium
fragility under climate shocks. Companion code to the technical appendix
*Design Validation for a Triple-Difference Study of Greenium Fragility:
A Monte Carlo Exercise* (C. Gan, 2026), prepared for a WRDTP Advanced
Quantitative Methods application.

The simulation calibrates the data-generating process to the exact sample
dimensions of the proposed study (150 matched green–conventional bond pairs,
800 trading days, 40 regional climate shocks, 12 regions, SFDR Level-2 regime
switch at the sample midpoint) and evaluates bias, RMSE, coverage, size and
statistical power of the proposed estimators, including a confounded-DGP
scenario that demonstrates how the dynamic event-study diagnostic flags an
EU-specific confounding trend while the pre-determined ownership-intensity
specification is immune to it.

## Reproduce

```bash
pip install -r requirements.txt
python run_all.py
```

`run_all.py` executes all 11 scenarios (500 replications each), the dynamic
event-study module (200 replications on scenarios S1, S4, W4), and builds all
tables (`output/tableM*.csv`) and figures (`output/figM*.png`). Fixed master
seeds make every number exactly reproducible; per-replication draws in
`output/raw/` are regenerated deterministically (excluded from git). Runtime
on an Apple-silicon laptop: about 5 minutes in total (~63 ms per replication
for the binary-regime scenarios with three estimators, ~18 ms for the
intensity scenarios).

## Structure

```
src/dgp.py             data-generating process, scenario matrix S1-S6 / W1-W6
src/estimators.py      TWFE within estimator with hand-built issuer-clustered
                       sandwich SEs; entropy balancing; CEM; event-study DDD
src/simulate.py        replication loop (incremental .npz output)
src/tables_figures.py  Tables M1-M4 and Figures M1-M3
output/raw/            per-scenario replication draws (regenerated)
```

## Estimators

| ID  | Estimator |
|-----|-----------|
| E1  | TWFE triple-difference, binary regime R1 = EU × post-Level-2 |
| EB  | E1 on entropy-balanced sample |
| CEM | E1 on CEM-matched sample |
| E6  | TWFE with continuous pre-determined intensity R2 (Article 8/9 ownership share) |

All estimators are implemented in pure NumPy (within transformation +
issuer-clustered sandwich); no statsmodels/linearmodels dependency.
Heterogeneity-robust DiD estimators (Callaway–Sant'Anna; Borusyak–Jaravel–
Spiess) are discussed in the appendix as a planned extension; treatment
timing here is simultaneous (single regime date), so TWFE bias from
staggered adoption does not arise by construction.
