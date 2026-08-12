#!/usr/bin/env python
"""
gate_simulation.py -- TASK C: the analytic model of the promotion gate, and the
Monte-Carlo study behind figures/fig_gate_acceptance.{pdf,png}.

The derivation this implements is in analysis/gate_model.md. In brief:

  Pool P = M u C, built by prepare_pd_df.py:37-77
    M : every reviewed misclassification report. The SQL condition is
        `model_predicted_x IS DISTINCT FROM correct_x`, so the incumbent is
        wrong on every row of M *by construction*.
    C : n_correct = int(|M| * r) rows sampled from complaints carrying no
        misclassification report, labelled with `c.<label>` -- which
        routers/complaints.py:63-86 wrote from the incumbent's own output, so
        the incumbent is right on every row of C *by construction*.

  => the incumbent's measured ACCURACY on P is exactly r/(1+r), for any true
     accuracy q and any admin reporting rate d. At r = 0.5 that is 1/3.

The gate compares macro-F1 rather than accuracy, and does so on a finite test
split, so this script simulates the actual comparison
(model_pipeline.py:400-447) instead of asserting the closed form for F1.

Outputs:
  results/gate_model.json
  figures/fig_gate_acceptance.pdf / .png

Usage: PYTHONIOENCODING=utf-8 python gate_simulation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

EVIDENCE = Path(__file__).resolve().parent.parent
RESULTS = EVIDENCE / "results"
FIGS = EVIDENCE / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# ---- pipeline constants, all read from the repository -----------------------
R_DEFAULT = 0.5        # prepare_pd_df.py:9   correct_ratio
TAU = 0.001            # configs.py:42        decision_threshold
DEPLOYED_SAMPLE = 300  # configs.py:41        deployed_sample_size
K = 3                  # urgency label space

# ---- palette (dataviz skill reference instance, validated light mode) -------
C_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # r = 0.25, 0.5, 1.0, 2.0
C_FAILOPEN = "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


# ===========================================================================
# Analytic part
# ===========================================================================
def incumbent_measured_accuracy(r: float) -> float:
    """Exact: the incumbent's measured accuracy on the retraining pool."""
    return r / (1.0 + r)


def acceptance_condition(r: float, a_C: float, tau: float = TAU) -> float:
    """Minimum accuracy on the FLAGGED-ERROR group a challenger needs to be promoted.

    Accept  <=>  (a_M + r*a_C)/(1+r) > r/(1+r) + tau
            <=>  a_M > r*(1 - a_C) + tau*(1+r)
    """
    return r * (1.0 - a_C) + tau * (1.0 + r)


def label_noise_in_C(q: float, d: float) -> float:
    """Fraction of the pseudo-labelled group whose stored label is actually wrong.

    Among complaints with no misclassification report, incumbent errors survive
    at rate (1-q)(1-d); correct predictions at rate q.
    """
    return (1 - q) * (1 - d) / (q + (1 - q) * (1 - d))


# ===========================================================================
# Monte-Carlo of the real comparison
# ===========================================================================
def simulate_pool(rng, n_mis: int, r: float, q: float, d: float, delta: float,
                  lam: float = 0.0, K: int = K):
    """One retraining pool + one incumbent/challenger prediction pair.

    Returns (stored_label, incumbent_pred, challenger_pred) over the pool.

    lam : degree to which the challenger mimics the stored label on group C
          (it was trained on those labels). lam = 0 is the CONSERVATIVE choice --
          it makes promotion harder, so the acceptance rates reported are lower
          bounds on what the real loop would produce.
    """
    n_c = int(n_mis * r)
    q_prime = np.clip(q + delta, 0.0, 1.0)

    def wrong_of(y):
        """A uniformly random label != y."""
        off = rng.integers(1, K, size=len(y))
        return (y + off) % K

    # --- group M: reviewed misclassifications ------------------------------
    yM = rng.integers(0, K, size=n_mis)          # true label (admin-corrected)
    incM = wrong_of(yM)                          # wrong by the SQL condition
    storedM = yM

    # --- group C: unflagged complaints, labelled by the incumbent ----------
    yC = rng.integers(0, K, size=n_c)
    # P(incumbent correct | not reported)
    q_unrep = q / (q + (1 - q) * (1 - d)) if (q + (1 - q) * (1 - d)) > 0 else 1.0
    okC = rng.random(n_c) < q_unrep
    incC = np.where(okC, yC, wrong_of(yC))
    storedC = incC                               # the incumbent's own output

    # --- challenger: true accuracy q' on TRUE labels ------------------------
    def chal(y):
        hit = rng.random(len(y)) < q_prime
        return np.where(hit, y, wrong_of(y))

    chalM = chal(yM)
    chalC = chal(yC)
    if lam > 0:                                   # partial mimicry of stored labels
        mimic = rng.random(n_c) < lam
        chalC = np.where(mimic, storedC, chalC)

    stored = np.concatenate([storedM, storedC])
    inc = np.concatenate([incM, incC])
    ch = np.concatenate([chalM, chalC])
    return stored, inc, ch


def one_trial(rng, n_pool: int, r: float, q: float, d: float, delta: float,
              lam: float, fail_open: bool) -> bool:
    """Reproduce model_pipeline.py:400-447 on one simulated pool. Returns accepted?"""
    n_mis = max(1, int(round(n_pool / (1 + r))))
    stored, inc, ch = simulate_pool(rng, n_mis, r, q, d, delta, lam)

    # The gate runs on the pool's TEST split (10% under the pipeline's own split).
    n = len(stored)
    idx = rng.permutation(n)
    n_test = max(K + 1, int(round(0.10 * n)))
    test = idx[:n_test]

    # challenger: macro-F1 over the whole test split
    f1_chal = f1_score(stored[test], ch[test], average="macro", zero_division=0)

    if fail_open:
        deployed = 0.0            # all endpoint queries failed, or api_endpoint unset
    else:
        # incumbent: macro-F1 over min(300, len(test)) rows of a shuffle of the same split
        m = min(DEPLOYED_SAMPLE, n_test)
        sub = test[rng.permutation(n_test)][:m]
        deployed = f1_score(stored[sub], inc[sub], average="macro", zero_division=0)

    return bool(f1_chal > deployed + TAU)


def acceptance_probability(r, q, d, delta, lam=0.0, n_pool=1500, trials=200,
                           fail_open=False, seed=0) -> float:
    rng = np.random.default_rng(seed)
    return float(np.mean([one_trial(rng, n_pool, r, q, d, delta, lam, fail_open)
                          for _ in range(trials)]))


# ===========================================================================
def main() -> None:
    out: dict = {
        "constants": {"correct_ratio_default": R_DEFAULT, "decision_threshold": TAU,
                      "deployed_sample_size": DEPLOYED_SAMPLE, "n_classes": K},
        "observability": {
            "r": "OBSERVABLE -- prepare_pd_df.py:9, default correct_ratio=0.5",
            "tau": "OBSERVABLE -- configs.py:42, and echoed in every auto-deploy commit message",
            "deployed_sample_size": "OBSERVABLE -- configs.py:41",
            "q (incumbent true accuracy on deployment traffic)": "NOT ESTABLISHED",
            "d (fraction of incumbent errors admins report)": "NOT ESTABLISHED",
            "a_M, a_C (challenger accuracy per pool group)": "NOT ESTABLISHED per promotion; simulated",
        },
    }

    # ---- analytic table ----------------------------------------------------
    print("=" * 92)
    print("ANALYTIC: incumbent's measured accuracy on the pool, and what a challenger needs")
    print("=" * 92)
    rows = []
    for r in (0.25, 0.5, 1.0, 2.0):
        a_inc = incumbent_measured_accuracy(r)
        rec = {"r": r, "incumbent_measured_accuracy": round(a_inc, 6),
               "min_a_M_if_a_C_1.00": round(acceptance_condition(r, 1.00), 6),
               "min_a_M_if_a_C_0.95": round(acceptance_condition(r, 0.95), 6),
               "min_a_M_if_a_C_0.90": round(acceptance_condition(r, 0.90), 6)}
        rows.append(rec)
        print(f"  r={r:<5} A_inc={a_inc:.4f}   required a_M: "
              f"a_C=1.00 -> {rec['min_a_M_if_a_C_1.00']:.4f} | "
              f"a_C=0.95 -> {rec['min_a_M_if_a_C_0.95']:.4f} | "
              f"a_C=0.90 -> {rec['min_a_M_if_a_C_0.90']:.4f}")
    out["analytic"] = rows
    out["operating_point"] = {
        "r": R_DEFAULT,
        "incumbent_measured_accuracy": incumbent_measured_accuracy(R_DEFAULT),
        "required_challenger_accuracy_on_flagged_errors_if_a_C_is_1":
            acceptance_condition(R_DEFAULT, 1.0),
        "reading": ("At the shipped operating point r=0.5 the incumbent scores exactly 1/3 on "
                    "the pool no matter how good it is. A challenger that merely reproduces the "
                    "incumbent on the pseudo-labelled third needs to get 0.15% of the incumbent's "
                    "known errors right to be promoted."),
    }
    print(f"\n  OPERATING POINT r=0.5: incumbent measured accuracy = "
          f"{incumbent_measured_accuracy(0.5):.4f}; a challenger with a_C=1 needs "
          f"a_M > {acceptance_condition(0.5, 1.0):.4f}")

    # ---- label noise in the pseudo-labelled group -------------------------
    noise = {}
    for q in (0.70, 0.85, 0.95):
        for d in (0.0, 0.25, 0.5, 1.0):
            noise[f"q={q},d={d}"] = round(label_noise_in_C(q, d), 6)
    out["label_noise_in_group_C"] = noise
    print(f"\n  label noise in group C at q=0.85: "
          + ", ".join(f"d={d} -> {label_noise_in_C(0.85, d):.3f}" for d in (0.0, 0.25, 0.5, 1.0)))

    # ---- Monte-Carlo -------------------------------------------------------
    print()
    print("=" * 92)
    print("MONTE-CARLO of the implemented comparison (macro-F1, finite test split)")
    print("=" * 92)
    deltas = np.round(np.arange(-0.30, 0.1001, 0.02), 4)
    r_values = [0.25, 0.5, 1.0, 2.0]
    q_main = 0.85
    TRIALS = 300

    curves = {}
    for i, r in enumerate(r_values):
        ys = [acceptance_probability(r, q_main, d=0.5, delta=float(dl), trials=TRIALS,
                                     seed=1000 + i * 97 + j)
              for j, dl in enumerate(deltas)]
        curves[str(r)] = ys
        print(f"  r={r:<5} P(accept): delta=-0.30 -> {ys[0]:.3f} | "
              f"delta=-0.10 -> {ys[list(deltas).index(-0.10)]:.3f} | "
              f"delta=0.00 -> {ys[list(deltas).index(0.0)]:.3f}")
    fail_curve = [1.0] * len(deltas)

    # q-sensitivity at a fixed, clearly-worse challenger
    q_grid = np.round(np.arange(0.50, 0.9901, 0.02), 4)
    delta_fixed = -0.10
    q_curves = {}
    for i, r in enumerate(r_values):
        q_curves[str(r)] = [acceptance_probability(r, float(q), d=0.5, delta=delta_fixed,
                                                   trials=TRIALS, seed=5000 + i * 89 + j)
                            for j, q in enumerate(q_grid)]

    out["monte_carlo"] = {
        "trials_per_point": TRIALS, "q_main": q_main, "d_assumed": 0.5,
        "lambda_challenger_mimicry": 0.0,
        "lambda_note": ("0 = the challenger is modelled as independent of the incumbent on the "
                        "pseudo-labelled group. This is the conservative choice: training on those "
                        "labels would raise a_C and make promotion easier, so the acceptance "
                        "probabilities reported here are lower bounds."),
        "n_pool": 1500,
        "deltas": [float(x) for x in deltas],
        "acceptance_by_r": curves,
        "fail_open_curve": fail_curve,
        "q_grid": [float(x) for x in q_grid],
        "delta_fixed_for_q_sweep": delta_fixed,
        "acceptance_vs_q_by_r": q_curves,
    }

    # widest band of delta<0 that is still accepted with probability >= 0.5
    bands = {}
    for r, ys in curves.items():
        neg = [(dl, y) for dl, y in zip(deltas, ys) if dl < 0]
        worst = min([dl for dl, y in neg if y >= 0.5], default=None)
        bands[r] = float(worst) if worst is not None else None
    out["monte_carlo"]["worst_delta_still_accepted_p50"] = bands
    print(f"\n  worst delta still accepted with P>=0.5, by r: {bands}")

    # =======================================================================
    # FIGURE
    # =======================================================================
    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "text.color": INK, "axes.labelcolor": INK, "figure.dpi": 150,
        "xtick.color": INK2, "ytick.color": INK2, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
    })
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9),
                                  gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("white")

    # Curves that sit exactly on P=1 everywhere would hide one another. Draw them
    # widest-first so each remains visible as a concentric strip; never offset the
    # data to separate them.
    flat = [r for r in r_values if all(v >= 0.999 for v in curves[str(r)])]
    widths = {r: (3.4 if r in flat and i == 0 else 1.8 if r in flat else 2.1)
              for i, r in enumerate(r_values)}
    if len(flat) >= 2:
        widths[flat[0]], widths[flat[-1]] = 3.4, 1.6

    def draw(a, xs, ys_by_r, flat_y):
        a.plot(xs, flat_y, color=C_FAILOPEN, lw=6.0, alpha=0.26, solid_capstyle="butt",
               zorder=3)
        a.plot(xs, flat_y, color=C_FAILOPEN, lw=1.9, ls=(0, (4, 2)), zorder=8)
        for i, r in enumerate(r_values):
            a.plot(xs, ys_by_r[str(r)], color=C_SERIES[i], lw=widths[r],
                   solid_capstyle="round", zorder=4 + i)

    # ---- panel A ----------------------------------------------------------
    ax.axhspan(0, 1.06, xmin=0, xmax=(0 - (deltas[0] - 0.005)) / (0.115 - (deltas[0] - 0.005)),
               color="#f4f3f0", zorder=0)
    draw(ax, deltas, curves, fail_curve)
    ax.axvline(0, color="#b9b8b2", lw=0.9, zorder=2)

    ax.text(deltas[0] + 0.004, 0.045, "challenger is genuinely WORSE  $\\rightarrow$  still promoted",
            color=INK2, fontsize=7.8, style="italic", zorder=9, ha="left")
    if len(flat) >= 2:
        lab = " and ".join([f"$r$ = {x}" for x in flat])
        ax.annotate(f"{lab} and the fail-open path\ncoincide at P = 1 across the whole range",
                    xy=(-0.155, 1.0), xytext=(-0.148, 0.72), fontsize=7.8, color=INK2,
                    ha="left", va="top", zorder=10,
                    arrowprops=dict(arrowstyle="-", color="#b9b8b2", lw=0.9,
                                    shrinkA=0, shrinkB=2))
    # selective direct labels, only where the curves are actually separated
    ax.text(-0.297, 0.79, "$r$ = 1.0", color=C_SERIES[2], fontsize=8, fontweight="bold",
            ha="left", va="bottom", zorder=10)
    ax.text(-0.188, 0.24, "$r$ = 2.0", color=C_SERIES[3], fontsize=8, fontweight="bold",
            ha="left", va="bottom", zorder=10)

    ax.set_xlim(deltas[0] - 0.005, 0.115)
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlabel(r"$\delta$  =  challenger true accuracy $-$ incumbent true accuracy")
    ax.set_ylabel("P(promoted)")
    ax.set_title("A.  The gate promotes models that are genuinely worse",
                 loc="left", color=INK)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.55, zorder=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- panel B ----------------------------------------------------------
    draw(ax2, q_grid, q_curves, [1.0] * len(q_grid))
    ax2.set_xlim(0.49, 1.0)
    ax2.set_ylim(-0.03, 1.06)
    ax2.set_xlabel(r"$q$  =  incumbent's true accuracy   (NOT ESTABLISHED)")
    ax2.set_ylabel("P(promoted)")
    ax2.set_title(r"B.  Barely sensitive to $q$   (at $\delta = -0.10$)", loc="left", color=INK)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.grid(axis="y", color=GRID, lw=0.6, alpha=0.55, zorder=1)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    r05_min_q06 = min(y for q, y in zip(q_grid, q_curves["0.5"]) if q >= 0.6)
    ax2.text(0.995, 0.055,
             "a challenger 10 points worse is promoted\n"
             f"with probability $\\geq$ {r05_min_q06:.2f} at the shipped $r$ = 0.5\n"
             "for every incumbent with $q \\geq 0.6$",
             color=INK2, fontsize=7.6, style="italic", va="bottom", ha="right", zorder=10)

    # ---- one shared legend ------------------------------------------------
    handles = [plt.Line2D([], [], color=C_SERIES[i], lw=2.4,
                          label=f"$r$ = {r}" + ("  (shipped default)" if r == R_DEFAULT else ""))
               for i, r in enumerate(r_values)]
    handles.append(plt.Line2D([], [], color=C_FAILOPEN, lw=2.0, ls=(0, (4, 2)),
                              label="fail-open path"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.10), handlelength=2.4, columnspacing=1.9)

    fig.text(0.008, -0.175,
             r"Gate: accept iff macro-F1$_{\rm challenger}$ > macro-F1$_{\rm incumbent}$ + 0.001 "
             r"(model_pipeline.py:447), both scored on the test split of a pool that is "
             r"$1/(1{+}r)$ flagged incumbent errors and $r/(1{+}r)$ rows labelled by the "
             "incumbent itself.",
             fontsize=7.2, color=INK2)
    fig.text(0.008, -0.225,
             "Monte-Carlo, 300 trials/point, pool n=1500, 3 classes, admin reporting rate d=0.5, "
             "challenger modelled as independent of the incumbent (conservative). "
             "Fail-open: model_pipeline.py:425-446.",
             fontsize=7.2, color=INK2)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_gate_acceptance.{ext}", facecolor="white",
                    bbox_inches="tight", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"\nWrote {FIGS / 'fig_gate_acceptance.pdf'} and .png")

    (RESULTS / "gate_model.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {RESULTS / 'gate_model.json'}")


if __name__ == "__main__":
    main()
