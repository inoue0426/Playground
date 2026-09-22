# Spatial niche world model — minimum experiment

## Protocol and data

Protocol A ran using public GEO GSE284989 MC38 Visium samples: IgG control and anti-PD-1-treated tumors. The dataset contains expression matrices and Visium array coordinates. The model test is machinery-only; it is not an interventional causal result.

Loaded 22 samples (11 control, 11 treated), 58,287 total spots, 20,551 source genes, and 2,000 global log-normalized HVGs. Each sample used a spatial kNN graph with k=6.

## Held-out split

Entire treated samples were held out for evaluation; no held-out expression was used for fitting or selection.

| Split | Samples |
|---|---|
| Held-out treated | GSM8694502_MC38_aPD1_NR_m10, GSM8694507_MC38_aPD1_R_m15_sec1, GSM8694511_MC38_aPD1_R_m16_sec3 |
| Training | 19 samples |

Within each held-out sample, the evaluation niche was a fixed contiguous 20% spatial block. Training used analogous fixed blocks in training samples. The conditioned model is a two-layer GCN with a treated/control embedding; the unconditional GCN removes that embedding; the MLP has no graph aggregation; nearest-niche copies the nearest unmasked spot.

## Metrics

| Model | MSE | MAE | Cosine | DE-gene Jaccard |
|---|---:|---:|---:|---:|
| graph_conditioned | 2.476482 | 1.317824 | 0.465102 | 0.022332 |
| graph_unconditional | 2.522260 | 1.326846 | 0.377009 | 0.020533 |
| mlp_no_graph | 2.560123 | 1.340199 | 0.400293 | 0.024041 |
| nearest_niche | 0.971284 | 0.678398 | 0.813569 | 0.241962 |

DE-gene Jaccard uses the top 100 absolute mean expression shifts versus unmasked spots, computed separately per held-out sample/model.

## Pre-registered decision

**NO-GO.** GO requires graph+condition to beat every baseline on MSE, MAE, and cosine. It did not meet the criterion; this is a final NO-GO and no variants were run.

Interpretation: this tests whether spatial graph aggregation plus an observed treatment label improves masked-niche reconstruction on held-out treated samples. It does not establish that the model predicts a causal drug intervention, because the learned task is reconstruction from tissue expression and observed condition, not a counterfactual treatment simulation.
