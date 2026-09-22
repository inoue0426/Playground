"""Run the minimum spatial niche world-model experiment (Protocol A).

This is a machinery test.  It evaluates whether a small graph model can
reconstruct masked spatial expression while receiving a treated/control
condition.  Held-out treated samples are never used for fitting or model
selection.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn


SEED = 67
K_NEIGHBORS = 6
HVG_COUNT = 2000
MASK_FRACTION = 0.20
HIDDEN_WIDTH = 32
EPOCHS = 15
LEARNING_RATE = 0.002
DE_GENE_COUNT = 100


@dataclass
class Sample:
    """One Visium sample with normalized expression and spatial metadata."""

    sample_id: str
    condition: int
    barcodes: np.ndarray
    expression: np.ndarray
    coordinates: np.ndarray


def set_seed(seed: int) -> None:
    """Set deterministic random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def decode(values: np.ndarray) -> np.ndarray:
    """Decode byte-valued HDF5 strings to ordinary strings."""
    return np.asarray([value.decode() if isinstance(value, bytes) else str(value) for value in values])


def load_h5(path: Path, hvg_indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a 10x filtered HDF5 matrix as spots-by-genes counts."""
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        shape = tuple(int(value) for value in matrix["shape"][:])
        counts = sp.csc_matrix(
            (matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]),
            shape=shape,
        )
        barcodes = decode(matrix["barcodes"][:])
        genes = decode(matrix["features"]["name"][:])
    if hvg_indices is not None:
        counts = counts[hvg_indices, :]
        genes = genes[hvg_indices]
    return counts.T.tocsr(), barcodes, genes


def normalize_counts(counts: sp.spmatrix) -> sp.csr_matrix:
    """Library-size normalize counts and apply log1p."""
    totals = np.asarray(counts.sum(axis=1)).ravel()
    scale = np.divide(1e4, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
    normalized = sp.diags(scale) @ counts
    normalized.data = np.log1p(normalized.data)
    return normalized.tocsr()


def sample_metadata(path: Path) -> tuple[str, int]:
    """Extract sample ID and treatment condition from a matrix filename."""
    name = path.name.replace("_spatial_filtered_feature_bc_matrix.h5", "")
    condition = int("_aPD1_" in name)
    return name, condition


def positions_for(matrix_path: Path, barcodes: np.ndarray) -> np.ndarray:
    """Load Visium array coordinates aligned to matrix barcodes."""
    position_path = matrix_path.name.replace(
        "_spatial_filtered_feature_bc_matrix.h5", "_spatial_tissue_positions_list.csv.gz"
    )
    frame = pd.read_csv(matrix_path.parent / position_path, header=None, compression="gzip")
    frame.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    frame = frame.set_index("barcode").reindex(barcodes)
    if frame.isna().any().any():
        raise ValueError(f"Coordinate/barcode mismatch in {matrix_path.name}")
    return frame[["array_row", "array_col"]].to_numpy(dtype=np.float32)


def discover_files(data_dir: Path) -> list[Path]:
    """Find all downloaded filtered Visium matrices."""
    files = sorted(data_dir.glob("*_spatial_filtered_feature_bc_matrix.h5"))
    if not files:
        raise FileNotFoundError("No filtered Visium HDF5 matrices found")
    return files


def select_hvgs(files: list[Path], count: int) -> tuple[np.ndarray, list[str]]:
    """Select highly variable genes using global log-normalized moments."""
    total_sum = None
    total_sq = None
    total_n = 0
    genes: list[str] = []
    for path in files:
        counts, _, current_genes = load_h5(path)
        normalized = normalize_counts(counts)
        if total_sum is None:
            total_sum = np.zeros(normalized.shape[1], dtype=np.float64)
            total_sq = np.zeros(normalized.shape[1], dtype=np.float64)
            genes = current_genes.tolist()
        total_sum += np.asarray(normalized.sum(axis=0)).ravel()
        total_sq += np.asarray(normalized.power(2).sum(axis=0)).ravel()
        total_n += normalized.shape[0]
    mean = total_sum / total_n
    variance = np.maximum(total_sq / total_n - mean**2, 0.0)
    indices = np.argsort(variance)[-min(count, len(variance)) :]
    return np.sort(indices), genes


def load_samples(files: list[Path], hvg_indices: np.ndarray) -> list[Sample]:
    """Load, normalize, and align all samples on the chosen HVGs."""
    samples = []
    for path in files:
        counts, barcodes, _ = load_h5(path, hvg_indices)
        normalized = normalize_counts(counts).toarray().astype(np.float32)
        sample_id, condition = sample_metadata(path)
        coordinates = positions_for(path, barcodes)
        samples.append(Sample(sample_id, condition, barcodes, normalized, coordinates))
    return samples


def knn_indices(coordinates: np.ndarray, k: int) -> np.ndarray:
    """Build a spatial kNN index using exact pairwise distances."""
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    distance = np.einsum("ijk,ijk->ij", deltas, deltas)
    np.fill_diagonal(distance, np.inf)
    return np.argpartition(distance, kth=k - 1, axis=1)[:, :k]


def contiguous_mask(coordinates: np.ndarray, fraction: float) -> np.ndarray:
    """Select a contiguous circular spatial block with the requested size."""
    center = np.median(coordinates, axis=0)
    distance = np.sum((coordinates - center) ** 2, axis=1)
    count = max(1, int(round(len(coordinates) * fraction)))
    mask = np.zeros(len(coordinates), dtype=bool)
    mask[np.argsort(distance)[:count]] = True
    return mask


def aggregate(features: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
    """Apply self-plus-six-neighbor mean aggregation."""
    neighbor_values = features[neighbors]
    return torch.cat((features.unsqueeze(1), neighbor_values), dim=1).mean(dim=1)


class GraphConditioned(nn.Module):
    """Two-layer graph encoder/decoder conditioned on treatment."""

    def __init__(self, genes: int, hidden: int, use_condition: bool) -> None:
        super().__init__()
        self.use_condition = use_condition
        condition_width = 4 if use_condition else 0
        self.condition = nn.Sequential(nn.Linear(1, 4), nn.ReLU()) if use_condition else None
        self.first = nn.Linear(genes + condition_width, hidden)
        self.second = nn.Linear(hidden, genes)

    def forward(
        self,
        features: torch.Tensor,
        neighbors: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Predict expression from one graph convolution and one decoder layer."""
        aggregated = aggregate(features, neighbors)
        if self.condition is not None:
            aggregated = torch.cat((aggregated, self.condition(condition)), dim=1)
        hidden = torch.relu(self.first(aggregated))
        hidden = aggregate(hidden, neighbors)
        return self.second(hidden)


class SpotMLP(nn.Module):
    """Non-graph baseline using masked spot expression, coordinates, and condition."""

    def __init__(self, genes: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(genes + 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, genes),
        )

    def forward(self, features: torch.Tensor, coordinates: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Predict expression without neighborhood aggregation."""
        return self.network(torch.cat((features, coordinates, condition), dim=1))


def build_training_arrays(
    samples: list[Sample], heldout: set[str], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create masked training arrays and graph neighbor offsets."""
    rng = np.random.default_rng(seed)
    feature_blocks = []
    target_blocks = []
    condition_blocks = []
    coordinate_blocks = []
    mask_blocks = []
    neighbor_blocks = []
    offset = 0
    for sample in samples:
        if sample.sample_id in heldout:
            continue
        mask = contiguous_mask(sample.coordinates, MASK_FRACTION)
        observed = sample.expression.copy()
        observed[mask] = 0.0
        neighbors = knn_indices(sample.coordinates, K_NEIGHBORS) + offset
        feature_blocks.append(observed)
        target_blocks.append(sample.expression)
        condition_blocks.append(np.full((len(sample.expression), 1), sample.condition, dtype=np.float32))
        mins = sample.coordinates.min(axis=0)
        spans = np.maximum(sample.coordinates.max(axis=0) - mins, 1.0)
        coordinate_blocks.append(((sample.coordinates - mins) / spans).astype(np.float32))
        mask_blocks.append(mask)
        neighbor_blocks.append(neighbors)
        offset += len(sample.expression)
    return (
        np.concatenate(feature_blocks),
        np.concatenate(target_blocks),
        np.concatenate(condition_blocks),
        np.concatenate(coordinate_blocks),
        np.concatenate(mask_blocks),
        np.concatenate(neighbor_blocks),
    )


def fit_model(
    model: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    conditions: np.ndarray,
    coordinates: np.ndarray,
    masked: np.ndarray,
    neighbors: np.ndarray | None,
    epochs: int,
) -> nn.Module:
    """Fit one fixed-configuration model on training masked spots only."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x = torch.from_numpy(features)
    y = torch.from_numpy(targets)
    c = torch.from_numpy(conditions)
    xy = torch.from_numpy(coordinates)
    mask = torch.from_numpy(masked)
    neighbor_tensor = torch.from_numpy(neighbors) if neighbors is not None else None
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        if isinstance(model, GraphConditioned):
            prediction = model(x, neighbor_tensor, c)
        else:
            prediction = model(x, xy, c)
        loss = nn.functional.mse_loss(prediction[mask], y[mask])
        loss.backward()
        optimizer.step()
    return model.eval()


def predict_sample(model: nn.Module, sample: Sample, masked: np.ndarray) -> np.ndarray:
    """Predict one masked sample while hiding target spot expression."""
    observed = sample.expression.copy()
    observed[masked] = 0.0
    mins = sample.coordinates.min(axis=0)
    spans = np.maximum(sample.coordinates.max(axis=0) - mins, 1.0)
    coordinates = ((sample.coordinates - mins) / spans).astype(np.float32)
    features = torch.from_numpy(observed)
    condition = torch.full((len(sample.expression), 1), sample.condition, dtype=torch.float32)
    if isinstance(model, GraphConditioned):
        neighbors = torch.from_numpy(knn_indices(sample.coordinates, K_NEIGHBORS))
        prediction = model(features, neighbors, condition)
    else:
        prediction = model(features, torch.from_numpy(coordinates), condition)
    return prediction.detach().numpy()


def nearest_prediction(sample: Sample, masked: np.ndarray) -> np.ndarray:
    """Predict each masked spot from its nearest observed spot."""
    observed_indices = np.flatnonzero(~masked)
    masked_indices = np.flatnonzero(masked)
    prediction = np.zeros_like(sample.expression)
    distances = sample.coordinates[masked_indices, None, :] - sample.coordinates[observed_indices][None, :, :]
    nearest = np.argmin(np.sum(distances**2, axis=2), axis=1)
    prediction[masked_indices] = sample.expression[observed_indices[nearest]]
    return prediction


def cosine_rows(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Compute row-wise cosine similarity."""
    numerator = np.sum(actual * predicted, axis=1)
    denominator = np.linalg.norm(actual, axis=1) * np.linalg.norm(predicted, axis=1)
    return numerator / np.maximum(denominator, 1e-8)


def de_jaccard(actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray) -> float:
    """Compare top absolute expression-shift gene sets against a local baseline."""
    actual_effect = np.abs(actual.mean(axis=0) - baseline.mean(axis=0))
    predicted_effect = np.abs(predicted.mean(axis=0) - baseline.mean(axis=0))
    actual_set = set(np.argsort(actual_effect)[-DE_GENE_COUNT:])
    predicted_set = set(np.argsort(predicted_effect)[-DE_GENE_COUNT:])
    return len(actual_set & predicted_set) / max(1, len(actual_set | predicted_set))


def main() -> None:
    """Run Protocol A once and write reproducible result artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    set_seed(SEED)
    files = discover_files(args.data)
    hvg_indices, all_genes = select_hvgs(files, HVG_COUNT)
    genes = [all_genes[index] for index in hvg_indices]
    samples = load_samples(files, hvg_indices)
    treated = sorted(sample.sample_id for sample in samples if sample.condition == 1)
    rng = random.Random(SEED)
    heldout = set(sorted(rng.sample(treated, 3)))
    train_samples = [sample for sample in samples if sample.sample_id not in heldout]
    test_samples = [sample for sample in samples if sample.sample_id in heldout]
    if not test_samples or any(sample.condition != 1 for sample in test_samples):
        raise RuntimeError("Held-out set must contain treated samples only")

    features, targets, conditions, coordinates, masked, neighbors = build_training_arrays(
        samples, heldout, SEED
    )
    gene_count = len(genes)
    conditioned = fit_model(
        GraphConditioned(gene_count, HIDDEN_WIDTH, True),
        features,
        targets,
        conditions,
        coordinates,
        masked,
        neighbors,
        EPOCHS,
    )
    unconditional = fit_model(
        GraphConditioned(gene_count, HIDDEN_WIDTH, False),
        features,
        targets,
        conditions,
        coordinates,
        masked,
        neighbors,
        EPOCHS,
    )
    mlp = fit_model(
        SpotMLP(gene_count, HIDDEN_WIDTH),
        features,
        targets,
        conditions,
        coordinates,
        masked,
        None,
        EPOCHS,
    )

    rows = []
    for sample in test_samples:
        test_mask = contiguous_mask(sample.coordinates, MASK_FRACTION)
        target = sample.expression[test_mask]
        baseline = sample.expression[~test_mask]
        predictions = {
            "graph_conditioned": predict_sample(conditioned, sample, test_mask)[test_mask],
            "graph_unconditional": predict_sample(unconditional, sample, test_mask)[test_mask],
            "mlp_no_graph": predict_sample(mlp, sample, test_mask)[test_mask],
            "nearest_niche": nearest_prediction(sample, test_mask)[test_mask],
        }
        for name, prediction in predictions.items():
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "condition": "aPD1_treated",
                    "masked_spots": int(test_mask.sum()),
                    "total_spots": len(test_mask),
                    "mse": float(np.mean((target - prediction) ** 2)),
                    "mae": float(np.mean(np.abs(target - prediction))),
                    "cosine": float(np.mean(cosine_rows(target, prediction))),
                    "de_gene_jaccard": float(de_jaccard(target, prediction, baseline)),
                    "model": name,
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(args.out_dir / "results.csv", index=False)
    summary = results.groupby("model")[["mse", "mae", "cosine", "de_gene_jaccard"]].mean().to_dict("index")
    main_model = summary["graph_conditioned"]
    decision = "GO" if all(
        main_model[metric] < summary[model][metric] if metric in ("mse", "mae") else main_model[metric] > summary[model][metric]
        for model in ("graph_unconditional", "mlp_no_graph", "nearest_niche")
        for metric in ("mse", "mae", "cosine")
    ) else "NO-GO"
    config = {
        "protocol": "A",
        "dataset": "GSE284989, MC38 Visium, IgG versus aPD1",
        "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE284989",
        "seed": SEED,
        "sample_count": len(samples),
        "control_sample_count": sum(sample.condition == 0 for sample in samples),
        "treated_sample_count": sum(sample.condition == 1 for sample in samples),
        "spots_total": int(sum(len(sample.expression) for sample in samples)),
        "genes_in_source_first_sample": len(all_genes),
        "hvg_count": gene_count,
        "k_neighbors": K_NEIGHBORS,
        "mask_fraction": MASK_FRACTION,
        "heldout_treated_samples": sorted(heldout),
        "train_samples": [sample.sample_id for sample in train_samples],
        "model": {"hidden_width": HIDDEN_WIDTH, "epochs": EPOCHS, "learning_rate": LEARNING_RATE},
        "summary_metrics": summary,
        "decision": decision,
        "de_gene_definition": f"top {DE_GENE_COUNT} absolute mean shifts versus unmasked spots; Jaccard",
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_report(args.out_dir / "results.md", config, results, summary, decision)
    print(json.dumps({"decision": decision, "summary": summary}, indent=2))


def write_report(path: Path, config: dict, results: pd.DataFrame, summary: dict, decision: str) -> None:
    """Write a concise protocol, metric, and decision report."""
    lines = [
        "# Spatial niche world model — minimum experiment",
        "",
        "## Protocol and data",
        "",
        "Protocol A ran using public GEO GSE284989 MC38 Visium samples: IgG control and anti-PD-1-treated tumors. The dataset contains expression matrices and Visium array coordinates. The model test is machinery-only; it is not an interventional causal result.",
        "",
        f"Loaded {config['sample_count']} samples ({config['control_sample_count']} control, {config['treated_sample_count']} treated), {config['spots_total']:,} total spots, {config['genes_in_source_first_sample']:,} source genes, and {config['hvg_count']:,} global log-normalized HVGs. Each sample used a spatial kNN graph with k={config['k_neighbors']}.",
        "",
        "## Held-out split",
        "",
        "Entire treated samples were held out for evaluation; no held-out expression was used for fitting or selection.",
        "",
        "| Split | Samples |",
        "|---|---|",
        f"| Held-out treated | {', '.join(config['heldout_treated_samples'])} |",
        f"| Training | {len(config['train_samples'])} samples |",
        "",
        "Within each held-out sample, the evaluation niche was a fixed contiguous 20% spatial block. Training used analogous fixed blocks in training samples. The conditioned model is a two-layer GCN with a treated/control embedding; the unconditional GCN removes that embedding; the MLP has no graph aggregation; nearest-niche copies the nearest unmasked spot.",
        "",
        "## Metrics",
        "",
        "| Model | MSE | MAE | Cosine | DE-gene Jaccard |",
        "|---|---:|---:|---:|---:|",
    ]
    order = ["graph_conditioned", "graph_unconditional", "mlp_no_graph", "nearest_niche"]
    for model in order:
        values = summary[model]
        lines.append(
            f"| {model} | {values['mse']:.6f} | {values['mae']:.6f} | {values['cosine']:.6f} | {values['de_gene_jaccard']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"DE-gene Jaccard uses the top {DE_GENE_COUNT} absolute mean expression shifts versus unmasked spots, computed separately per held-out sample/model.",
            "",
            "## Pre-registered decision",
            "",
            f"**{decision}.** GO requires graph+condition to beat every baseline on MSE, MAE, and cosine. "
            + ("It met the criterion." if decision == "GO" else "It did not meet the criterion; this is a final NO-GO and no variants were run."),
            "",
            "Interpretation: this tests whether spatial graph aggregation plus an observed treatment label improves masked-niche reconstruction on held-out treated samples. It does not establish that the model predicts a causal drug intervention, because the learned task is reconstruction from tissue expression and observed condition, not a counterfactual treatment simulation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
