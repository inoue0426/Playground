# DrugCombo minimum compositional experiment

## Data gate and missingness

The corrected gate passed: every drug appears in at least 20 observed pairs (minimum 22), and missingness is diffuse. The largest cell-line deficit is 159 pairs (4.3% of missing endpoints). The workbook contains 17,901 / 21,571 endpoints on the observed 583 x 37 grid. It has 22 fully paired drugs (37 pairs each) and 16 partially paired drugs (22 pairs each).

The source has no monotherapy or dose columns. Drug effect vectors are therefore training-only combination-sensitivity profile proxies; a held-out drug gets the training cell-line mean fallback. Dose-scaling is not estimable and was not fabricated.

## Compositional split

Seed: 69. Held-out fully paired drugs: AZD1775, ZOLINZA. Held-out partially paired drugs: DEXAMETHASONE, SN-38.
Train pairs: 470; test pairs: 113 (pairs touching held-out fully paired drugs: 73; touching held-out partially paired drugs: 44).
All pairs touching a held-out drug were evaluation-only; no test target was used for fitting or model selection.

| Held-out group | Drugs | Observed pairs/drug | Test pairs touched |
|---|---|---:|---:|
| Fully paired | AZD1775, ZOLINZA | 37 each | 73 |
| Partially paired | DEXAMETHASONE, SN-38 | 22 each | 44 |

## Missingness audit

Per-drug observed endpoint counts / observed pair counts:

| Drug | Endpoints | Pairs |
|---|---:|---:|
| 5-FU | 704 | 22 |
| ABT-888 | 1161 | 37 |
| AZD1775 | 1218 | 37 |
| BEZ-235 | 1204 | 37 |
| BORTEZOMIB | 562 | 37 |
| CARBOPLATIN | 715 | 22 |
| CYCLOPHOSPHAMIDE | 677 | 22 |
| DASATINIB | 1228 | 37 |
| DEXAMETHASONE | 679 | 22 |
| DINACICLIB | 859 | 37 |
| DOXORUBICIN | 750 | 22 |
| ERLOTINIB | 1186 | 37 |
| ETOPOSIDE | 706 | 22 |
| GELDANAMYCIN | 1082 | 37 |
| GEMCITABINE | 762 | 22 |
| L778123 | 1244 | 37 |
| LAPATINIB | 1143 | 37 |
| METFORMIN | 698 | 22 |
| METHOTREXATE | 673 | 22 |
| MITOMYCINE | 711 | 22 |
| MK-2206 | 1204 | 37 |
| MK-4541 | 1149 | 37 |
| MK-4827 | 1190 | 37 |
| MK-5108 | 1205 | 37 |
| MK-8669 | 1207 | 37 |
| MK-8776 | 1131 | 37 |
| MRK-003 | 1198 | 37 |
| OXALIPLATIN | 662 | 22 |
| PACLITAXEL | 583 | 22 |
| PD325901 | 1191 | 37 |
| SN-38 | 708 | 22 |
| SORAFENIB | 958 | 37 |
| SUNITINIB | 1258 | 37 |
| TEMOZOLOMIDE | 1163 | 37 |
| TOPOTECAN | 763 | 22 |
| VINBLASTINE | 607 | 22 |
| VINORELBINE | 501 | 22 |
| ZOLINZA | 1162 | 37 |

Per-cell-line observed endpoints / missing pairs on the 583-pair grid:

| Cell line | Observed endpoints | Missing pairs |
|---|---:|---:|
| A2058 | 502 | 81 |
| A2780 | 508 | 75 |
| A375 | 456 | 127 |
| A427 | 461 | 122 |
| CAOV3 | 479 | 104 |
| COLO320DM | 503 | 80 |
| DLD1 | 494 | 89 |
| EFM192B | 518 | 65 |
| ES2 | 478 | 105 |
| HCT116 | 424 | 159 |
| HT144 | 489 | 94 |
| HT29 | 488 | 95 |
| KPL1 | 539 | 44 |
| LNCAP | 466 | 117 |
| LOVO | 490 | 93 |
| MDAMB436 | 505 | 78 |
| MSTO | 476 | 107 |
| NCIH1650 | 536 | 47 |
| NCIH2122 | 485 | 98 |
| NCIH23 | 446 | 137 |
| NCIH460 | 460 | 123 |
| NCIH520 | 434 | 149 |
| OV90 | 501 | 82 |
| OVCAR3 | 430 | 153 |
| PA1 | 453 | 130 |
| RKO | 478 | 105 |
| RPMI7951 | 490 | 93 |
| SKMEL30 | 478 | 105 |
| SKMES1 | 495 | 88 |
| SKOV3 | 485 | 98 |
| SW620 | 445 | 138 |
| SW837 | 482 | 101 |
| T47D | 520 | 63 |
| UACC62 | 484 | 99 |
| UWB1289 | 512 | 71 |
| VCAP | 490 | 93 |
| ZR751 | 521 | 62 |

## Held-out metrics

| Baseline | MSE | MAE |
|---|---:|---:|
| learned_operator | 73.989777 | 6.859768 |
| additive | 544.650879 | 21.768003 |
| bliss | 353.369720 | 17.205462 |
| nearest_pair | 110.475510 | 7.853723 |

## Algebraic probes

- Learned-operator commutativity mean absolute violation: 0.00000000.
- Additive commutativity mean absolute violation: 0.00000000.
- Dose-scaling monotonicity: not estimable because the source workbook contains no dose measurements.

## Pre-registered decision

**GO.** The learned operator must beat both additive and Bliss on both MSE and MAE and have commutativity no worse than additive. It met these criteria.

Interpretation: this minimum experiment evaluates compositional extrapolation to pairs touching four entirely held-out drugs, using only the incomplete workbook and fixed training configuration. The dose and monotherapy limitations mean the result tests a training-derived sensitivity-profile proxy rather than a true dose-aware monotherapy composition law.
