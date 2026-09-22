"""Run the preregistered minimum DrugCombo compositional experiment.

The supplied workbook contains aggregated combination sensitivity values but
does not contain monotherapy or dose columns.  This script therefore builds
training-only sensitivity-profile proxies for drugs and uses a fixed-dose MLP
operator.  It never uses held-out pair targets for fitting or model choice.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


SEED = 69
HOLDOUT_FULL_COUNT = 2
HOLDOUT_PARTIAL_COUNT = 2
EPOCHS = 250
LEARNING_RATE = 0.003
HIDDEN_WIDTH = 32
TARGET_COLUMN = "Senstivity"


def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def canonical_pair(drug_a: str, drug_b: str) -> tuple[str, str]:
    """Return a deterministic unordered drug-pair representation."""
    return tuple(sorted((str(drug_a), str(drug_b))))


def add_pair_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a canonical pair column without modifying the input frame."""
    result = frame.copy()
    result["pair"] = [
        "||".join(canonical_pair(a, b))
        for a, b in zip(result["Drug1"], result["Drug2"])
    ]
    return result


def verify_gate(frame: pd.DataFrame) -> dict:
    """Verify the corrected data gate and return its audit information."""
    pairs = sorted(frame["pair"].unique())
    cells = sorted(frame["Cell_Line"].unique())
    drugs = sorted(set(frame["Drug1"]) | set(frame["Drug2"]))
    pair_counts = {}
    endpoint_counts = {}
    for drug in drugs:
        mask = (frame["Drug1"] == drug) | (frame["Drug2"] == drug)
        pair_counts[drug] = int(frame.loc[mask, "pair"].nunique())
        endpoint_counts[drug] = int(mask.sum())
    cell_counts = frame.groupby("Cell_Line").size().reindex(cells, fill_value=0)
    cell_missing = {cell: int(len(pairs) - cell_counts[cell]) for cell in cells}
    missing_total = int(sum(cell_missing.values()))
    max_cell_fraction = max(cell_missing.values()) / missing_total
    fully_paired = sorted(drug for drug, count in pair_counts.items() if count == len(cells))
    partially_paired = sorted(
        drug for drug, count in pair_counts.items() if count != len(cells)
    )
    gate = {
        "min_pairs_per_drug": min(pair_counts.values()),
        "min_pairs_threshold": 20,
        "pairs_gate_pass": min(pair_counts.values()) >= 20,
        "max_cell_missing_fraction": max_cell_fraction,
        "missingness_gate_pass": max_cell_fraction <= 0.5,
        "fully_paired_drugs": fully_paired,
        "partially_paired_drugs": partially_paired,
        "observed_endpoints": int(len(frame)),
        "grid_endpoints": int(len(pairs) * len(cells)),
        "missing_endpoints_on_583x37_grid": missing_total,
        "per_drug_pair_counts": pair_counts,
        "per_drug_endpoint_counts": endpoint_counts,
        "per_cell_endpoint_counts": {cell: int(cell_counts[cell]) for cell in cells},
        "per_cell_missing_pair_counts": cell_missing,
    }
    if not gate["pairs_gate_pass"] or not gate["missingness_gate_pass"]:
        raise RuntimeError(f"Corrected data gate failed: {gate}")
    return gate


def choose_split(frame: pd.DataFrame, seed: int) -> dict:
    """Choose whole held-out drugs from both coverage groups."""
    rng = random.Random(seed)
    pairs = set(frame["pair"])
    drugs = sorted(set(frame["Drug1"]) | set(frame["Drug2"]))
    cells = frame["Cell_Line"].nunique()
    pair_counts = {
        drug: int(
            frame.loc[(frame["Drug1"] == drug) | (frame["Drug2"] == drug), "pair"].nunique()
        )
        for drug in drugs
    }
    fully = [drug for drug in drugs if pair_counts[drug] == cells]
    partial = [drug for drug in drugs if pair_counts[drug] != cells]
    heldout_full = sorted(rng.sample(fully, HOLDOUT_FULL_COUNT))
    heldout_partial = sorted(rng.sample(partial, HOLDOUT_PARTIAL_COUNT))
    heldout_drugs = sorted(heldout_full + heldout_partial)
    test_pairs = sorted(
        pair for pair in pairs if any(drug in pair.split("||") for drug in heldout_drugs)
    )
    train_pairs = sorted(pairs - set(test_pairs))
    return {
        "seed": seed,
        "heldout_full_drugs": heldout_full,
        "heldout_partial_drugs": heldout_partial,
        "heldout_drugs": heldout_drugs,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "train_pair_count": len(train_pairs),
        "test_pair_count": len(test_pairs),
        "test_pairs_with_full_drug": sum(
            any(drug in pair.split("||") for drug in heldout_full) for pair in test_pairs
        ),
        "test_pairs_with_partial_drug": sum(
            any(drug in pair.split("||") for drug in heldout_partial) for pair in test_pairs
        ),
    }


def make_effect_profiles(
    train: pd.DataFrame, all_drugs: list[str], cells: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Build training-only drug effect profiles with cell-wise fallback values."""
    cell_means = train.groupby("Cell_Line")[TARGET_COLUMN].mean().reindex(cells)
    global_mean = float(train[TARGET_COLUMN].mean())
    fallback = cell_means.fillna(global_mean).to_numpy(dtype=np.float32)
    profiles: dict[str, np.ndarray] = {}
    for drug in all_drugs:
        rows = train[(train["Drug1"] == drug) | (train["Drug2"] == drug)]
        if rows.empty:
            profiles[drug] = fallback.copy()
        else:
            profiles[drug] = (
                rows.groupby("Cell_Line")[TARGET_COLUMN]
                .mean()
                .reindex(cells)
                .fillna(pd.Series(fallback, index=cells))
                .to_numpy(dtype=np.float32)
            )
    return profiles, {cell: float(value) for cell, value in zip(cells, fallback)}


def pair_feature(
    drug_a: str, drug_b: str, profiles: dict[str, np.ndarray]
) -> np.ndarray:
    """Create a symmetric composition feature from two drug profiles."""
    first = profiles[drug_a]
    second = profiles[drug_b]
    return np.concatenate(((first + second) / 2.0, np.abs(first - second))).astype(
        np.float32
    )


class CompositionMLP(nn.Module):
    """Small symmetric MLP mapping two effect profiles to a cell-line vector."""

    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_width, HIDDEN_WIDTH),
            nn.ReLU(),
            nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH),
            nn.ReLU(),
            nn.Linear(HIDDEN_WIDTH, output_width),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict one combination response vector per input row."""
        return self.network(features)


class FeatureScaler:
    """Small dependency-free standard scaler for fixed feature matrices."""

    def fit(self, features: np.ndarray) -> "FeatureScaler":
        """Fit column means and standard deviations."""
        self.mean_ = features.mean(axis=0)
        self.scale_ = features.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Standardize features using the training statistics."""
        return (features - self.mean_) / self.scale_


def fit_operator(
    train: pd.DataFrame,
    train_pairs: list[str],
    profiles: dict[str, np.ndarray],
    cells: list[str],
) -> tuple[CompositionMLP, FeatureScaler]:
    """Fit the fixed-config MLP on training pairs only."""
    targets = fill_missing_columns(response_matrix(train, train_pairs, cells))
    targets = targets.astype(np.float32, copy=True)
    features = np.stack(
        [pair_feature(*pair.split("||"), profiles) for pair in train_pairs]
    )
    scaler = FeatureScaler().fit(features)
    scaled = scaler.transform(features).astype(np.float32)
    model = CompositionMLP(scaled.shape[1], len(cells))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x_tensor = torch.from_numpy(scaled)
    y_tensor = torch.from_numpy(targets)
    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()
    return model.eval(), scaler


def response_matrix(frame: pd.DataFrame, pairs: list[str], cells: list[str]) -> np.ndarray:
    """Return pair-by-cell response values for complete observed pairs."""
    return (
        frame.pivot_table(index="pair", columns="Cell_Line", values=TARGET_COLUMN)
        .reindex(index=pairs, columns=cells)
        .to_numpy(dtype=np.float32)
    )


def fill_missing_columns(matrix: np.ndarray) -> np.ndarray:
    """Fill training-only missing cells with the corresponding column mean."""
    result = matrix.copy()
    column_means = np.nanmean(result, axis=0)
    rows, columns = np.where(np.isnan(result))
    result[rows, columns] = column_means[columns]
    return result


def baseline_predictions(
    train: pd.DataFrame,
    test_pairs: list[str],
    train_pairs: list[str],
    profiles: dict[str, np.ndarray],
    cells: list[str],
) -> dict[str, np.ndarray]:
    """Generate additive, Bliss, and nearest-training-pair predictions."""
    additive = []
    bliss = []
    for pair in test_pairs:
        first, second = pair.split("||")
        a = profiles[first]
        b = profiles[second]
        additive.append(a + b)
        bliss.append(100.0 * ((a / 100.0) + (b / 100.0) - (a / 100.0) * (b / 100.0)))

    train_features = np.stack(
        [pair_feature(*pair.split("||"), profiles) for pair in train_pairs]
    )
    train_norm = train_features / np.maximum(
        np.linalg.norm(train_features, axis=1, keepdims=True), 1e-8
    )
    train_matrix = fill_missing_columns(response_matrix(train, train_pairs, cells))
    nearest = []
    for pair in test_pairs:
        feature = pair_feature(*pair.split("||"), profiles)
        normalized = feature / max(float(np.linalg.norm(feature)), 1e-8)
        index = int(np.argmax(train_norm @ normalized))
        nearest.append(train_matrix[index])
    return {
        "additive": np.asarray(additive),
        "bliss": np.asarray(bliss),
        "nearest_pair": np.asarray(nearest),
    }


def pair_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute mean-over-pairs MSE and MAE across cell-line endpoints."""
    observed = np.isfinite(actual)
    squared_error = np.where(observed, (actual - predicted) ** 2, np.nan)
    absolute_error = np.where(observed, np.abs(actual - predicted), np.nan)
    per_pair_mse = np.nanmean(squared_error, axis=1)
    per_pair_mae = np.nanmean(absolute_error, axis=1)
    return {
        "mse": float(per_pair_mse.mean()),
        "mae": float(per_pair_mae.mean()),
        "per_pair_mse": per_pair_mse.tolist(),
        "per_pair_mae": per_pair_mae.tolist(),
    }


def main() -> None:
    """Run the gate, train the operator, score baselines, and write artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/oneil_raw.xlsx"))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    set_seed(SEED)
    out_dir = args.out_dir
    frame = add_pair_column(pd.read_excel(args.data))
    gate = verify_gate(frame)
    cells = sorted(frame["Cell_Line"].unique())
    all_drugs = sorted(set(frame["Drug1"]) | set(frame["Drug2"]))
    split = choose_split(frame, SEED)
    train = frame[frame["pair"].isin(split["train_pairs"])].copy()
    test = frame[frame["pair"].isin(split["test_pairs"])].copy()
    profiles, fallback = make_effect_profiles(train, all_drugs, cells)
    model, scaler = fit_operator(train, split["train_pairs"], profiles, cells)

    test_actual = response_matrix(test, split["test_pairs"], cells)
    test_features = np.stack(
        [pair_feature(*pair.split("||"), profiles) for pair in split["test_pairs"]]
    )
    learned = model(torch.from_numpy(scaler.transform(test_features).astype(np.float32))).detach().numpy()
    predictions = baseline_predictions(
        train, split["test_pairs"], split["train_pairs"], profiles, cells
    )
    predictions["learned_operator"] = learned

    metrics = {}
    for name, prediction in predictions.items():
        scores = pair_metrics(test_actual, prediction)
        metrics[name] = {key: value for key, value in scores.items() if not key.startswith("per_pair")}

    swapped = np.stack(
        [pair_feature(*reversed(pair.split("||")), profiles) for pair in split["test_pairs"]]
    )
    swapped_pred = model(torch.from_numpy(scaler.transform(swapped).astype(np.float32))).detach().numpy()
    commutativity = np.abs(learned - swapped_pred).mean(axis=1)
    additive_swapped = np.stack(
        [profiles[pair.split("||")[0]] + profiles[pair.split("||")[1]] for pair in split["test_pairs"]]
    )
    additive_commutativity = np.abs(predictions["additive"] - additive_swapped).mean(axis=1)
    algebraic = {
        "learned_operator_commutativity_mean_abs": float(commutativity.mean()),
        "learned_operator_commutativity_max_abs": float(commutativity.max()),
        "additive_commutativity_mean_abs": float(additive_commutativity.mean()),
        "dose_scaling_violation_rate": None,
        "dose_scaling_status": "not_estimable: source workbook has no dose columns",
    }
    learned_better = all(
        metrics["learned_operator"][metric] < metrics["additive"][metric]
        and metrics["learned_operator"][metric] < metrics["bliss"][metric]
        for metric in ("mse", "mae")
    )
    comm_ok = algebraic["learned_operator_commutativity_mean_abs"] <= algebraic[
        "additive_commutativity_mean_abs"
    ] + 1e-12
    decision = "GO" if learned_better and comm_ok else "NO-GO"

    rows = []
    for i, pair in enumerate(split["test_pairs"]):
        first, second = pair.split("||")
        for j, cell in enumerate(cells):
            actual = float(test_actual[i, j])
            row = {
                "pair": pair,
                "drug_a": first,
                "drug_b": second,
                "cell_line": cell,
                "actual_sensitivity": actual,
            }
            for name, prediction in predictions.items():
                value = float(prediction[i, j])
                row[f"{name}_prediction"] = value
                row[f"{name}_absolute_error"] = abs(value - actual)
                row[f"{name}_squared_error"] = (value - actual) ** 2
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)

    config = {
        "data_source": "data/oneil_raw.xlsx; public O'Neil-derived workbook",
        "target": TARGET_COLUMN,
        "seed": SEED,
        "gate": gate,
        "split": split,
        "model": {
            "architecture": "symmetric MLP over mean and absolute difference of training-only effect profiles",
            "hidden_width": HIDDEN_WIDTH,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "dose_inputs": "unavailable in source; fixed-dose operator",
            "effect_profile_fallback": "training cell-line mean for drugs absent from training pairs",
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_results_markdown(out_dir / "results.md", gate, split, metrics, algebraic, decision)
    print(json.dumps({"decision": decision, "metrics": metrics, "algebraic": algebraic}, indent=2))


def write_results_markdown(
    path: Path,
    gate: dict,
    split: dict,
    metrics: dict,
    algebraic: dict,
    decision: str,
) -> None:
    """Write the auditable results report."""
    lines = [
        "# DrugCombo minimum compositional experiment",
        "",
        "## Data gate and missingness",
        "",
        "The corrected gate passed: every drug appears in at least 20 observed pairs (minimum 22), and missingness is diffuse. The largest cell-line deficit is "
        f"{max(gate['per_cell_missing_pair_counts'].values())} pairs ({gate['max_cell_missing_fraction']:.1%} of missing endpoints). "
        f"The workbook contains {gate['observed_endpoints']:,} / {gate['grid_endpoints']:,} endpoints on the observed 583 x 37 grid. "
        "It has 22 fully paired drugs (37 pairs each) and 16 partially paired drugs (22 pairs each).",
        "",
        "The source has no monotherapy or dose columns. Drug effect vectors are therefore training-only combination-sensitivity profile proxies; a held-out drug gets the training cell-line mean fallback. Dose-scaling is not estimable and was not fabricated.",
        "",
        "## Compositional split",
        "",
        f"Seed: {split['seed']}. Held-out fully paired drugs: {', '.join(split['heldout_full_drugs'])}. Held-out partially paired drugs: {', '.join(split['heldout_partial_drugs'])}.",
        f"Train pairs: {split['train_pair_count']}; test pairs: {split['test_pair_count']} (pairs touching held-out fully paired drugs: {split['test_pairs_with_full_drug']}; touching held-out partially paired drugs: {split['test_pairs_with_partial_drug']}).",
        "All pairs touching a held-out drug were evaluation-only; no test target was used for fitting or model selection.",
        "",
        "| Held-out group | Drugs | Observed pairs/drug | Test pairs touched |",
        "|---|---|---:|---:|",
        f"| Fully paired | {', '.join(split['heldout_full_drugs'])} | 37 each | {split['test_pairs_with_full_drug']} |",
        f"| Partially paired | {', '.join(split['heldout_partial_drugs'])} | 22 each | {split['test_pairs_with_partial_drug']} |",
        "",
        "## Missingness audit",
        "",
        "Per-drug observed endpoint counts / observed pair counts:",
        "",
        "| Drug | Endpoints | Pairs |",
        "|---|---:|---:|",
    ]
    for drug in sorted(gate["per_drug_pair_counts"]):
        lines.append(
            f"| {drug} | {gate['per_drug_endpoint_counts'][drug]} | {gate['per_drug_pair_counts'][drug]} |"
        )
    lines.extend(
        [
            "",
            "Per-cell-line observed endpoints / missing pairs on the 583-pair grid:",
            "",
            "| Cell line | Observed endpoints | Missing pairs |",
            "|---|---:|---:|",
        ]
    )
    for cell in sorted(gate["per_cell_endpoint_counts"]):
        lines.append(
            f"| {cell} | {gate['per_cell_endpoint_counts'][cell]} | {gate['per_cell_missing_pair_counts'][cell]} |"
        )
    lines.extend(
        [
            "",
        "## Held-out metrics",
        "",
        "| Baseline | MSE | MAE |",
        "|---|---:|---:|",
        ]
    )
    for name in ("learned_operator", "additive", "bliss", "nearest_pair"):
        lines.append(f"| {name} | {metrics[name]['mse']:.6f} | {metrics[name]['mae']:.6f} |")
    lines.extend(
        [
            "",
            "## Algebraic probes",
            "",
            f"- Learned-operator commutativity mean absolute violation: {algebraic['learned_operator_commutativity_mean_abs']:.8f}.",
            f"- Additive commutativity mean absolute violation: {algebraic['additive_commutativity_mean_abs']:.8f}.",
            "- Dose-scaling monotonicity: not estimable because the source workbook contains no dose measurements.",
            "",
            "## Pre-registered decision",
            "",
            f"**{decision}.** The learned operator must beat both additive and Bliss on both MSE and MAE and have commutativity no worse than additive. "
            + ("It met these criteria." if decision == "GO" else "It did not meet all these criteria; the negative result is final and no variants were run."),
            "",
            "Interpretation: this minimum experiment evaluates compositional extrapolation to pairs touching four entirely held-out drugs, using only the incomplete workbook and fixed training configuration. The dose and monotherapy limitations mean the result tests a training-derived sensitivity-profile proxy rather than a true dose-aware monotherapy composition law.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
