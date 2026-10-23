## Main table: halueval

| Detector | AUC [95% CI] | Within-model AUC [95% CI] | AUC − length [95% CI] | Within-model − length [95% CI] | Beats length (within-model) | p50 ms (range of runs) | p95 ms | Precision / recall / F1 | Frontier |
|---|---|---|---|---|---|---|---|---|---|
| Lexical overlap | 0.917 [0.904, 0.929] | – | -0.057 [-0.072, -0.043] | – | – | 0.0225 (0.0207–0.0323) | 0.0404 | 0.95 / 0.94 / 0.94 | frontier |
| DistilBERT (HaluEval) | 0.997 [0.994, 0.999] | – | +0.023 [+0.018, +0.028] | – | – | 7.61 (7.43–7.84) | 11.2 | 0.99 / 0.99 / 0.99 | frontier |
| NLI | 0.527 [0.507, 0.547] | – | -0.447 [-0.469, -0.426] | – | – | 29.6 (28.6–29.8) | 34.9 | 0.50 / 1.00 / 0.67 | dominated by Lexical overlap, DistilBERT (HaluEval) |
| Length (reference) | 0.974 [0.968, 0.979] | – | – | – | – | 0.00134 (0.00116–0.00165) | 0.00255 | 0.96 / 0.93 / 0.95 | reference |

## Main table: ragtruth_qa

| Detector | AUC [95% CI] | Within-model AUC [95% CI] | AUC − length [95% CI] | Within-model − length [95% CI] | Beats length (within-model) | p50 ms (range of runs) | p95 ms | Precision / recall / F1 | Frontier |
|---|---|---|---|---|---|---|---|---|---|
| Lexical overlap | 0.745 [0.702, 0.785] | 0.672 [0.618, 0.720] | +0.032 [-0.027, +0.090] | +0.041 [-0.042, +0.117] | no | 0.129 (0.112–0.175) | 0.279 | 0.29 / 0.78 / 0.42 | frontier |
| DistilBERT transfer | 0.434 [0.378, 0.489] | 0.464 [0.406, 0.523] | -0.278 [-0.364, -0.190] | -0.167 [-0.264, -0.069] | no | 9.2 (8.76–10.1) | 14.7 | 0.18 / 0.94 / 0.30 | dominated by Lexical overlap |
| DistilBERT in-domain | 0.788 [0.746, 0.829] | 0.699 [0.641, 0.753] | +0.076 [+0.036, +0.115] | +0.068 [+0.017, +0.119] | yes | 9.28 (8.61–10.9) | 16 | 0.32 / 0.76 / 0.45 | frontier |
| NLI chunked | 0.666 [0.622, 0.708] | 0.588 [0.534, 0.639] | -0.047 [-0.109, +0.019] | -0.043 [-0.116, +0.036] | no | 164 (160–164) | 609 | 0.23 / 0.84 / 0.36 | dominated by DistilBERT in-domain |
| Length (reference) | 0.712 [0.665, 0.756] | 0.631 [0.573, 0.684] | – | – | – | 0.00723 (0.0053–0.0121) | 0.014 | 0.33 / 0.54 / 0.41 | reference |

## Latency

| Dataset | Detector | Runs | p50 ms, first 200 (range) | p50 ms, random 200 | Random vs first | p95 ms | p99 ms |
|---|---|---|---|---|---|---|---|
| halueval | NLI | 5 | 29.6 (28.6–29.8) | – | – | 34.9 | 40.4 |
| halueval | Length (reference) | 5 | 0.00134 (0.00116–0.00165) | – | – | 0.00255 | 0.00374 |
| halueval | DistilBERT (HaluEval) | 5 | 7.61 (7.43–7.84) | – | – | 11.2 | 15.1 |
| halueval | Lexical overlap | 5 | 0.0225 (0.0207–0.0323) | – | – | 0.0404 | 0.0634 |
| ragtruth_qa | DistilBERT transfer | 5 | 9.2 (8.76–10.1) | 9.18 | 0% | 14.7 | 19.6 |
| ragtruth_qa | Length (reference) | 5 | 0.00723 (0.0053–0.0121) | 0.00957 | +32% | 0.014 | 0.0231 |
| ragtruth_qa | NLI chunked | 5 | 164 (160–164) | 126 | -23% | 609 | 907 |
| ragtruth_qa | DistilBERT in-domain | 5 | 9.28 (8.61–10.9) | 8.91 | -4% | 16 | 21.9 |
| ragtruth_qa | Lexical overlap | 5 | 0.129 (0.112–0.175) | 0.154 | +20% | 0.279 | 0.311 |

## Truncation (test split)

| Dataset | Arm | Group | Pairs | Truncated | Context tokens median / p90 | Context kept (mean) |
|---|---|---|---|---|---|---|
| halueval | DistilBERT (HaluEval) | all | 3006 | 0 (0.0%) | 72 / 119 | 100.0% |
| halueval | DistilBERT (HaluEval) | positive | 1503 | 0 (0.0%) | 72 / 119 | 100.0% |
| halueval | DistilBERT (HaluEval) | negative | 1503 | 0 (0.0%) | 72 / 119 | 100.0% |
| ragtruth_qa | DistilBERT transfer | all | 900 | 284 (31.6%) | 291 / 449 | 92.7% |
| ragtruth_qa | DistilBERT transfer | positive | 160 | 68 (42.5%) | 305 / 440 | 88.3% |
| ragtruth_qa | DistilBERT transfer | negative | 740 | 216 (29.2%) | 291 / 451 | 93.7% |
| ragtruth_qa | DistilBERT in-domain | all | 900 | 284 (31.6%) | 291 / 449 | 92.7% |
| ragtruth_qa | DistilBERT in-domain | positive | 160 | 68 (42.5%) | 305 / 440 | 88.3% |
| ragtruth_qa | DistilBERT in-domain | negative | 740 | 216 (29.2%) | 291 / 451 | 93.7% |
