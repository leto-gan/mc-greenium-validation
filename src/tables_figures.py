"""Build Tables M1-M4 (CSV) and Figures M1-M3 (PNG) from output/raw/*.npz.

Usage (from repo root):  python src/tables_figures.py
Writes to output/: tableM1_calibration.csv, tableM2_bias.csv,
tableM3_power.csv, tableM4_size.csv, figM1_power.png, figM2_distribution.png,
figM3_eventstudy.png, summary.json (machine-readable stats for the appendix).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dgp import (SCENARIOS, EVENT_STUDY_SCENARIOS, TAU, KAPPA, MU_GREEN,
                 SD_ISSUER_BASE, SIGMA_TOTAL, REGION_CORR, RHO_AR,
                 N_ISSUERS, T_DAYS, K_SHOCKS, N_REGIONS, SHOCK_WINDOW,
                 DRIFT_BP_PER_YEAR, DELIST_SHARE)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "output" / "raw"
OUT = ROOT / "output"

EST_LABEL = {"E1": "TWFE-DDD", "EB": "EB + TWFE", "CEM": "CEM + TWFE",
             "E6": "TWFE intensity (R2)"}
SCEN_LABEL = {
    "S1": "S1 benchmark (phi=-4)", "S2": "S2 null (phi=0)",
    "S3": "S3 weak (phi=-2)", "S6": "S6 strong (phi=-6)",
    "S4": "S4 confounded (EU drift)", "S5": "S5 unbalanced (15% delisted)",
    "W1": "W1 intensity (phi=-4)", "W2": "W2 intensity null (phi=0)",
    "W3": "W3 intensity weak (phi=-2)", "W6": "W6 intensity strong (phi=-6)",
    "W4": "W4 intensity confounded",
    "S3b": "S3b mid (phi=-3)", "W3b": "W3b intensity mid (phi=-3)",
}


def load(name):
    d = np.load(RAW / f"{name}.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


CRIT = 1.976   # t(0.975, 149), cluster count minus one


def stats(phi_hat, phi_se, phi_true):
    bias = phi_hat.mean() - phi_true
    sd = phi_hat.std(ddof=1)
    rmse = np.sqrt(((phi_hat - phi_true) ** 2).mean())
    cover = np.mean(np.abs(phi_hat - phi_true) <= CRIT * phi_se)
    reject = np.mean(np.abs(phi_hat / phi_se) > CRIT)
    return dict(bias=float(bias), sd=float(sd), rmse=float(rmse),
                coverage=float(cover), reject=float(reject))


def main():
    summary = {"scenarios": {}, "power": {}, "size": {}, "eventstudy": {}}

    # ---- Table M1: calibration ------------------------------------------
    rows = [
        ("Matched green-conventional pairs", N_ISSUERS, "Proposal sample frame (RP 4.1.5)"),
        ("Trading days T", T_DAYS, "SFDR Level 1 -> 2 window"),
        ("Climate shocks K", K_SHOCKS, "EM-DAT/NOAA annual frequency"),
        ("Geographic regions", N_REGIONS, "Shock clustering"),
        ("Elevated-response window", f"{SHOCK_WINDOW} days", "—"),
        ("Baseline greenium (bp)", f"{MU_GREEN} (sd {SD_ISSUER_BASE})", "Zerbib (2019); Liu & Hu (2025)"),
        ("True differential shock effect tau (bp)", TAU, "Del Giudice et al. (2025)"),
        ("Common conventional shock response kappa (bp)", KAPPA, "Design assumption"),
        ("True moderation phi (bp)", "-4; {-2, -6} in scenarios", "50% buffering benchmark"),
        ("AR(1) rho / region correlation", f"{RHO_AR} / {REGION_CORR}", "—"),
        ("Daily residual sd (bp)", SIGMA_TOTAL, "Bond-level spread volatility"),
        ("S4 confounder", f"+{DRIFT_BP_PER_YEAR} bp/year EU drift post-2023", "Identification threat"),
        ("S5 delisting share", DELIST_SHARE, "Robustness"),
    ]
    with open(OUT / "tableM1_calibration.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Parameter", "Value", "Source/Calibration"])
        w.writerows(rows)

    # ---- Table M2: bias / SD / RMSE / coverage --------------------------
    order = ["S1", "S2", "S3", "S3b", "S6", "S4", "S5",
             "W1", "W2", "W3", "W3b", "W6", "W4"]
    with open(OUT / "tableM2_bias.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Scenario", "Estimator", "True phi (bp)", "Bias (bp)",
                    "SD", "RMSE", "95% CI coverage", "Rejection rate"])
        for name in order:
            d = load(name)
            phi_true = float(json.loads(d["meta"][()])["phi_true"])
            for est in (["E1", "EB", "CEM"] if name.startswith("S")
                        else ["E6"]):
                s = stats(d[f"{est}_phi_hat"], d[f"{est}_phi_se"], phi_true)
                summary["scenarios"].setdefault(name, {})[est] = s
                w.writerow([SCEN_LABEL[name], EST_LABEL[est], phi_true,
                            f"{s['bias']:.3f}", f"{s['sd']:.3f}",
                            f"{s['rmse']:.3f}", f"{s['coverage']:.3f}",
                            f"{s['reject']:.3f}"])

    # ---- Table M3: power -------------------------------------------------
    with open(OUT / "tableM3_power.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["True phi (bp)", "E1 TWFE-DDD", "EB+TWFE", "CEM+TWFE",
                    "E6 intensity"])
        for phi, sbin, wbin in [(-2, "S3", "W3"), (-3, "S3b", "W3b"),
                                (-4, "S1", "W1"), (-6, "S6", "W6")]:
            row = [phi]
            summary["power"][str(phi)] = {}
            for scen, est in [(sbin, "E1"), (sbin, "EB"), (sbin, "CEM"),
                              (wbin, "E6")]:
                r = summary["scenarios"][scen][est]["reject"]
                summary["power"][str(phi)][est] = r
                row.append(f"{r:.3f}")
            w.writerow(row)

    # ---- Table M4: size ---------------------------------------------------
    with open(OUT / "tableM4_size.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Null scenario", "Estimator", "Rejection rate (nominal 0.05)"])
        for scen, ests in [("S2", ["E1", "EB", "CEM"]), ("W2", ["E6"])]:
            for est in ests:
                r = summary["scenarios"][scen][est]["reject"]
                summary["size"][f"{scen}_{est}"] = r
                w.writerow([SCEN_LABEL[scen], EST_LABEL[est], f"{r:.3f}"])

    # ---- Figure M1: power curve ------------------------------------------
    phis = [2, 3, 4, 6]
    plt.figure(figsize=(6.4, 4.2))
    for est, marker in [("E1", "o"), ("EB", "s"), ("CEM", "^"), ("E6", "D")]:
        pw = [summary["power"][str(-p)][est] for p in phis]
        plt.plot(phis, pw, marker=marker, label=EST_LABEL[est])
    plt.axhline(0.80, ls="--", lw=1, color="grey")
    plt.text(2.05, 0.815, "power = 0.80", color="grey", fontsize=8)
    plt.xlabel("True moderating effect |phi| (bp)")
    plt.ylabel("Rejection rate at alpha = 0.05")
    plt.ylim(0, 1.05)
    plt.legend(frameon=False, fontsize=8)
    plt.title("Figure M1. Statistical power by true effect size")
    plt.tight_layout()
    plt.savefig(OUT / "figM1_power.png", dpi=200)
    plt.close()

    # ---- Figure M2: distribution of phi_hat under S1 ----------------------
    d = load("S1")
    plt.figure(figsize=(6.4, 4.2))
    plt.hist(d["E1_phi_hat"], bins=40, density=True, alpha=0.75,
             color="steelblue")
    plt.axvline(-4.0, color="black", lw=1.5, ls="--", label="true phi = -4 bp")
    plt.axvline(d["E1_phi_hat"].mean(), color="darkred", lw=1.2,
                label=f"MC mean = {d['E1_phi_hat'].mean():.2f} bp")
    plt.xlabel("Estimated phi (bp), TWFE-DDD under S1")
    plt.ylabel("Density")
    plt.legend(frameon=False, fontsize=8)
    plt.title("Figure M2. Sampling distribution of the DDD estimator (S1)")
    plt.tight_layout()
    plt.savefig(OUT / "figM2_distribution.png", dpi=200)
    plt.close()

    # ---- Figure M3: event-study diagnostic --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=False)
    for ax, names, title in [
        (axes[0], ["S1", "S4"], "Binary regime R1 = EU x post-Level-2"),
        (axes[1], ["W4"], "Continuous intensity R2 (Article 8/9 ownership)"),
    ]:
        for name, color in zip(names, ["steelblue", "darkred"]):
            d = np.load(RAW / f"eventstudy_{name}.npz")
            edges, coefs = d["bin_edges"], d["coefs"]
            centers = (edges[:-1] + edges[1:]) / 2
            mean = np.nanmean(coefs, axis=0)
            lo = np.nanpercentile(coefs, 5, axis=0)
            hi = np.nanpercentile(coefs, 95, axis=0)
            ax.plot(centers, mean, marker="o", ms=4, color=color,
                    label=SCEN_LABEL[name])
            ax.fill_between(centers, lo, hi, color=color, alpha=0.15)
        ax.axvline(-2.5, ls=":", lw=1, color="grey")
        ax.axhline(0.0, lw=0.8, color="black")
        ax.set_xlabel("Trading days relative to climate shock")
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        summary["eventstudy"][names[0]] = "see figM3_eventstudy.png"
    axes[0].set_ylabel("DDD coefficient (bp)")
    fig.suptitle("Figure M3. Dynamic event-study diagnosis of the confounded DGP",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "figM3_eventstudy.png", dpi=200)
    plt.close(fig)

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("tables and figures written to", OUT)


if __name__ == "__main__":
    main()
