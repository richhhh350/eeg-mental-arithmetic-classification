"""Source-only grouped epoch selection and full-source EEGNet refitting."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from .eegnet import ChannelScaler, EEGNet, infer, seed_all, train_epoch


def subject_ba(y, probabilities, groups):
    predicted = np.asarray(probabilities) >= 0.5
    result = {}
    for subject in sorted(np.unique(groups)):
        mask = groups == subject
        if set(y[mask]) != {0, 1}:
            raise ValueError("Both validation conditions required per participant")
        result[str(subject)] = float(np.mean([np.mean(predicted[mask & (y == c)] == c) for c in (0, 1)]))
    return result


def balanced_cells(y, groups):
    counts = [int(np.sum((groups == s) & (y == c))) for s in np.unique(groups) for c in (0, 1)]
    if not counts or min(counts) == 0 or len(set(counts)) != 1:
        raise ValueError("Frozen sampler requires equal nonzero subject-condition cell counts")
    return counts[0]


def _initialize(x, y, config, seed):
    spec = config["training"]
    seed_all(seed, spec["threads"])
    scaler = ChannelScaler().fit(x)
    tensor_x = torch.from_numpy(scaler.transform(x))
    tensor_y = torch.from_numpy(np.asarray(y, dtype=np.int64))
    model = EEGNet(**config["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    return scaler, tensor_x, tensor_y, model, optimizer, generator


def fit_source_eegnet(x, y, groups, config, seed, callback=None):
    """No target inputs/labels in this interface. Returns a fresh all-source fit."""
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    if x.ndim != 3 or not (len(x) == len(y) == len(groups)) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Invalid source arrays")
    balanced_cells(y, groups)
    spec = config["training"]
    if not (1 <= spec["min_epochs"] <= spec["max_epochs"] and spec["patience"] >= 1):
        raise ValueError("Invalid epoch budget")
    folds = []
    for index, (fit_indices, val_indices) in enumerate(GroupKFold(config["inner_folds"]).split(x, y, groups)):
        fit_subjects, val_subjects = sorted(set(groups[fit_indices])), sorted(set(groups[val_indices]))
        if set(fit_subjects) & set(val_subjects):
            raise AssertionError("Inner subject leakage")
        inner_seed = seed + 10000 * (index + 1)
        scaler, tensor_x, tensor_y, model, optimizer, generator = _initialize(x[fit_indices], y[fit_indices], config, inner_seed)
        validation_x = scaler.transform(x[val_indices])
        history, best_score, best_epoch = [], -1.0, 0
        for epoch in range(1, spec["max_epochs"] + 1):
            loss = train_epoch(model, optimizer, tensor_x, tensor_y, generator, spec["batch_size"])
            probabilities = infer(model, validation_x, spec["batch_size"])
            scores = subject_ba(y[val_indices], probabilities, groups[val_indices])
            score = float(np.mean(list(scores.values())))
            history.append({"epoch": epoch, "training_loss": loss, "mean_subject_validation_ba": score, "validation_subject_ba": scores})
            if score > best_score + 1e-12:
                best_score, best_epoch = score, epoch
            if epoch >= spec["min_epochs"] and epoch - best_epoch >= spec["patience"]:
                break
        folds.append({"fold": index, "seed": inner_seed, "fit_subjects": fit_subjects,
                      "validation_subjects": val_subjects, "best_epoch": best_epoch,
                      "best_mean_subject_ba": best_score, "epochs_run": len(history),
                      "hit_epoch_cap": len(history) == spec["max_epochs"], "history": history})
        if callback:
            callback({"phase": "inner", "fold": index, "epochs_run": len(history), "best_epoch": best_epoch})
    selected_epochs = int(np.median([f["best_epoch"] for f in folds]))
    scaler, tensor_x, tensor_y, model, optimizer, generator = _initialize(x, y, config, seed)
    final_history = []
    for epoch in range(1, selected_epochs + 1):
        loss = train_epoch(model, optimizer, tensor_x, tensor_y, generator, spec["batch_size"])
        final_history.append({"epoch": epoch, "training_loss": loss})
    model.eval()
    selection = {"seed": seed, "fit_subjects": sorted(set(groups)), "inner_folds": folds,
                 "selected_epochs": selected_epochs, "epoch_aggregation": "median_of_three_inner_best_epochs",
                 "final_history": final_history, "target_used_for_selection": False,
                 "parameters": sum(p.numel() for p in model.parameters()),
                 "standardization": "source-training channel mean/std, ddof0"}
    return model, scaler, selection
