# Representation over Routing: Diagnosing Temporal Routing Pathologies in Multi-Timescale PPO

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.13517-b31b1b.svg)](https://arxiv.org/abs/2604.13517)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2604.13517-blue.svg)](https://doi.org/10.48550/arXiv.2604.13517)
[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model-yellow?logo=huggingface)](https://huggingface.co/ben-dlwlrma/Representation-Over-Routing)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-yellow?logo=huggingface)](https://huggingface.co/spaces/ben-dlwlrma/Representation-Over-Routing-Demo)

This repository contains the code and lightweight diagnostic assets for the preprint **"Representation over Routing: Diagnosing Temporal Routing Pathologies in Multi-Timescale PPO"**.

The project is a mechanistic diagnostic study of temporal routing failure modes in multi-timescale PPO. The experiments use LunarLander-v2 as a controlled, visually interpretable sandbox rather than as evidence for broad benchmark superiority.

## Paper

arXiv: https://arxiv.org/abs/2604.13517

The paper PDF and source manuscript are distributed through arXiv. This code repository is not intended to be a manuscript archive.

The current manuscript frames the work around three points:

- Differentiable actor-side temporal routing can expose a scale-mismatch shortcut inside the PPO surrogate.
- Error-based gradient-free routing can bias the actor toward low-error short-horizon heads.
- Target Decoupling removes the actor-side routing pathway while keeping multi-horizon critic prediction as auxiliary regularization.

## Qualitative Rollouts

These rollouts show the four diagnostic stages using the released actor checkpoints. They are visual examples for inspecting behavior, not additional benchmark evidence.

| Baseline PPO | Differentiable Routing |
| --- | --- |
| ![Baseline PPO hovering rollout](docs/baseline_hovering.gif) | ![Differentiable routing crash rollout](docs/surrogate_hacking_crash.gif) |

| Error-Based Routing | Target Decoupling |
| --- | --- |
| ![Error-based routing wandering rollout](docs/temporal_paradox_wandering.gif) | ![Target Decoupling landing rollout](docs/target_decoupling_landing.gif) |

## Method Summary

The multi-timescale critic predicts values for:

```text
gamma = [0.5, 0.9, 0.99, 0.999]
```

The diagnostic variants test where these heads enter the policy update:

- **Baseline PPO:** single-horizon PPO reference.
- **Differentiable routing:** an actor-side attention router mixes multi-horizon advantages.
- **Error-based routing:** routing weights are computed from absolute temporal-difference errors.
- **Target Decoupling:** the actor uses only the long-horizon advantage, while auxiliary critic heads remain as regularizers.

Target Decoupling is not presented here as a general performance booster. In the reported run set, it acts as a structural separation principle: it removes an exploitable actor-side routing channel and improves the observed worst-seed return and dispersion relative to a strict long-horizon baseline.

## Repository Structure

Expected public repository layout:

```text
.
├── experiments/                          # Training, diagnostic, and ablation entrypoints
│   ├── 1_baseline.py
│   ├── 2_surrogate_hacking_attention.py
│   ├── 3_temporal_paradox_variance.py
│   ├── 4_target_decoupling_final.py
│   ├── 5_evaluate_seeds_plot.py
│   └── 6_ablation_auxiliary_variance.py
├── analysis/                             # TensorBoard readers and plotting scripts
│   ├── plot_surrogate_hacking_diagnostics.py
│   ├── plot_error_routing_diagnostic.py
│   ├── plot_and_test.py
│   └── plot_salvage_return_boxplot.py
├── scripts/
│   └── render/                           # Regenerate rendered stage rollouts
│       ├── record_1_baseline.py
│       ├── record_2_surrogate.py
│       ├── record_3_paradox.py
│       └── record_4_decoupling.py
├── checkpoints/                          # Small pretrained actor weights
│   ├── 1_baseline.pth
│   ├── 2_surrogate_hacking_attention.pth
│   ├── 3_temporal_paradox_variance.pth
│   └── 4_target_decoupling_final.pth
├── docs/                                 # Selected diagnostic figures
├── requirements.txt
└── README.md
```

Large experimental artifacts are handled separately. The paper source directory, LaTeX build artifacts, raw TensorBoard logs, local editor files, caches, and archived exploratory scripts are not required in the public GitHub package. The small pretrained actor weights are included here for convenience and are also distributed through the Hugging Face model repository linked above.

## Reproducing the Main Diagnostics

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the commands below from the repository root so that relative paths such as `runs/`, `docs/`, and `checkpoints/` resolve correctly.

Train or regenerate the differentiable-routing diagnostic:

```bash
python experiments/2_surrogate_hacking_attention.py
python analysis/plot_surrogate_hacking_diagnostics.py
```

Train or regenerate the error-based routing diagnostic. The plotting script expects the fixed run names shown below:

```bash
python experiments/3_temporal_paradox_variance.py --seed 1 --run_name error_routing_seed_1
python experiments/3_temporal_paradox_variance.py --seed 2 --run_name error_routing_seed_2
python experiments/3_temporal_paradox_variance.py --seed 3 --run_name error_routing_seed_3
python experiments/3_temporal_paradox_variance.py --seed 4 --run_name error_routing_seed_4
python experiments/3_temporal_paradox_variance.py --seed 5 --run_name error_routing_seed_5
```

Generate the error-based routing figure after those five TensorBoard runs are available locally:

```bash
python analysis/plot_error_routing_diagnostic.py
```

### Reproducibility: Error-Based Routing Runs

`analysis/plot_error_routing_diagnostic.py` uses a fixed set of five TensorBoard runs for the error-based routing diagnostic:

| Seed | Run directory |
| --- | --- |
| 1 | `runs/error_routing_seed_1` |
| 2 | `runs/error_routing_seed_2` |
| 3 | `runs/error_routing_seed_3` |
| 4 | `runs/error_routing_seed_4` |
| 5 | `runs/error_routing_seed_5` |

These runs use the controlled PPO configuration for `LunarLander-v2` and log `charts/episodic_return` plus the `weights/gamma_*` routing weights used by the diagnostic plot. Raw TensorBoard logs are not required in the public GitHub package; if they are absent, use the generated figure in `docs/` or regenerate the runs before calling the plotting script.

Run the auxiliary-head ablation and reliability plots:

```bash
python experiments/6_ablation_auxiliary_variance.py
python analysis/plot_and_test.py
python analysis/plot_salvage_return_boxplot.py
```

Run the baseline-vs-decoupling seed comparison:

```bash
python experiments/5_evaluate_seeds_plot.py
```

Regenerate rendered rollouts and actor weights:

```bash
python scripts/render/record_1_baseline.py
python scripts/render/record_2_surrogate.py
python scripts/render/record_3_paradox.py
python scripts/render/record_4_decoupling.py
```

## Figures

The repository may include selected generated figures for README display and quick inspection:

- `docs/surrogate_hacking_diagnostic_triad.png`: return, HackRate, and attention entropy for differentiable routing.
- `docs/error_routing_diagnostic.png`: return and routing weights for error-based routing.
- `docs/salvage_return_boxplot.png`: final-return reliability comparison for strict long-horizon PPO and Target Decoupling.

These figures are intended as diagnostic evidence in a controlled PPO setting. They should not be read as broad claims about all environments or all multi-timescale RL methods.

## Citation

```bibtex
@misc{sunRepresentationRoutingDiagnosing2026,
  title = {Representation over {{Routing}}: {{Diagnosing Temporal Routing Pathologies}} in {{Multi-Timescale PPO}}},
  shorttitle = {Representation over {{Routing}}},
  author = {Sun, Jing},
  year = 2026,
  publisher = {arXiv},
  doi = {10.48550/ARXIV.2604.13517},
  urldate = {2026-04-16},
  copyright = {Creative Commons Attribution 4.0 International},
  keywords = {Artificial Intelligence (cs.AI),FOS: Computer and information sciences,Machine Learning (cs.LG)}
}
```
