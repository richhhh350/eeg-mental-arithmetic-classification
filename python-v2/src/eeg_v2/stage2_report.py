"""Aggregate verified stage2 experiments and create exploratory analyses."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .cli import new_output
from .provenance import read_csv, read_json, write_csv, write_json
from .stage2 import experiment_specs, load_variant, verify_experiment
from .stage2_analysis import paired_descriptive, source_pca


LABELS = {
    "s2_de_lr": "DE / LR (reference)", "s2_de_lda": "DE / shrinkage LDA",
    "s2_de_svm": "DE / RBF SVM", "s2_dummy": "Dummy (prior)",
    "s2_relative_lr": "Relative band power / LR", "s2_hjorth_lr": "Hjorth / LR",
    "s2_de_hjorth_lr": "DE + Hjorth / LR", "s2_without_theta_lr": "DE without theta / LR",
    "s2_without_alpha_lr": "DE without alpha / LR", "s2_without_beta_lr": "DE without beta / LR",
    "s2_gamma_sensitivity_lr": "DE + gamma / LR (sensitivity)", "s2_window1_lr": "1-second DE / LR",
    "s2_window4_lr": "4-second DE / LR", "s2_overlap_train_lr": "2-second DE / LR, train overlap 50%",
    "s1_de_lr": "DE / LR", "s1_de_lda": "DE / shrinkage LDA", "s1_de_svm": "DE / RBF SVM",
    "s1_de_dummy": "Dummy (prior)",
}


def _plot_comparisons(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for protocol in ("S2", "S1"):
        subset = [r for r in rows if r["protocol"] == protocol]
        fig, ax = plt.subplots(figsize=(11, 7 if protocol == "S2" else 3.8), constrained_layout=True)
        values = [r["balanced_accuracy"] for r in subset]
        ypos = np.arange(len(subset))
        ax.barh(ypos, values, color=["#203f5e" if r["experiment"] == "s2_de_lr" else "#2d8390" for r in subset])
        for y, value in zip(ypos, values):
            ax.text(value + .008, y, f"{value:.1%}", va="center", fontsize=9)
        ax.set_yticks(ypos, [LABELS[r["experiment"]] for r in subset])
        ax.invert_yaxis()
        ax.axvline(.5, color="#af4e42", ls="--", lw=1.2, label="Dummy BA = 0.50")
        ax.set(xlim=(0, 1.08), xlabel="Equal-subject mean balanced accuracy",
               title=f"{protocol}: {'nested LOSO' if protocol == 'S2' else 'isolated later time blocks'} / 36 participants")
        ax.legend(loc="lower right")
        fig.savefig(output / f"{protocol.lower()}_comparison.png", dpi=180)
        plt.close(fig)


def analyze(prepared, runs, output, stage1_run):
    runs, stage1_run = Path(runs), Path(stage1_run)
    status = read_json(runs / "batch_status.json")
    if status["status"] != "complete":
        raise ValueError("Do not summarize a failed batch as complete")
    arrays, pools, config, manifest = load_variant(prepared, "de2")
    specs = experiment_specs(config)
    if status["experiments_requested"] != len(specs):
        raise ValueError("Full matrix required for final analysis")
    output = new_output(output)
    summaries, metrics_by_experiment = [], {}
    for spec in specs:
        folder = runs / spec["id"]
        verify_experiment(folder, prepared)
        summary = read_json(folder / "summary.json")
        if summary["requested_folds"] != 36 or summary["completed_folds"] != 36:
            raise ValueError("All 36 participants are required")
        rows = read_csv(folder / "subject_metrics.csv")
        metrics_by_experiment[spec["id"]] = rows
        summaries.append({"experiment": spec["id"], "protocol": spec["protocol"],
                          "model": spec["kind"], "variant": spec["variant"],
                          "n_features": manifest["variants"][spec["variant"]]["n_features"],
                          **{name: summary[f"mean_subject_{name}"] for name in
                             ("balanced_accuracy", "accuracy", "macro_f1", "positive_class_f1", "roc_auc")},
                          "min_subject_ba": summary["min_subject_ba"], "max_subject_ba": summary["max_subject_ba"],
                          "fit_windows_per_subject_model": int(rows[0]["fit_windows"]),
                          "test_windows_per_subject": int(rows[0]["test_windows"]),
                          "elapsed_seconds": summary["elapsed_seconds"]})
    write_csv(output / "experiment_summary.csv", summaries)
    reference = metrics_by_experiment["s2_de_lr"]
    comparisons = {spec["id"]: paired_descriptive(reference, metrics_by_experiment[spec["id"]],
                    seed=config["seed"], replicates=config["bootstrap_replicates"])
                   for spec in specs if spec["protocol"] == "S2" and spec["id"] != "s2_de_lr"}
    write_json(output / "paired_descriptive.json", comparisons)
    # Replay of the existing reference is an integration check, not a new seed run.
    old_rows = read_csv(stage1_run / "predictions.csv")
    new_rows = read_csv(runs / "s2_de_lr/predictions.csv")
    old = {r["window_id"]: float(r["probability_arithmetic"]) for r in old_rows}
    new = {r["window_id"]: float(r["score"]) for r in new_rows}
    if old.keys() != new.keys():
        raise AssertionError("Stage1 reference window set differs")
    difference = max(abs(old[key] - new[key]) for key in old)
    if difference > 1e-12:
        raise AssertionError(f"Reference predictions differ by {difference}")
    # Overlap changes only fitting pools, not inner validation or outer test windows.
    overlap, overlap_rows, _, _ = load_variant(prepared, "overlap2")
    overlap_eval_matches = np.array_equal(arrays["X_eval"], overlap["X_eval"]) and pools["eval"] == overlap_rows["eval"]
    if not overlap_eval_matches:
        raise AssertionError("Overlap ablation changed evaluation windows")
    pca_results = {}
    for target in config["pca_targets"]:
        pca_results[target] = source_pca(arrays["X_eval"], arrays["groups_eval"], arrays["y_eval"],
                                        target, output / f"pca_{target}", config["seed"])
    _plot_comparisons(summaries, output)
    write_json(output / "analysis.json", {"dataset_id": manifest["dataset_id"], "experiments": len(specs),
               "outer_evaluations": 36 * len(specs), "stage1_reference_max_probability_error": difference,
               "overlap_evaluation_features_and_metadata_identical": overlap_eval_matches,
               "pca_targets": list(pca_results), "statistical_scope": config["comparison_statistics"]})
    print(f"Verified and analyzed {len(specs)} experiments, {36*len(specs)} outer evaluations", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stage1-run", default="outputs/loso-stage1")
    analyze(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
