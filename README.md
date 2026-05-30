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
├── 1_baseline.py                         # Single-horizon PPO baseline
├── 2_surrogate_hacking_attention.py      # Differentiable temporal routing diagnostic
├── 3_temporal_paradox_variance.py        # Error-based routing diagnostic
├── 4_target_decoupling_final.py          # Target Decoupling PPO
├── 5_evaluate_seeds_plot.py              # Baseline vs Target Decoupling seed comparison
├── 6_ablation_auxiliary_variance.py      # Auxiliary-head weighting ablation
├── plot_surrogate_hacking_diagnostics.py # Three-panel routing diagnostic plot
├── plot_error_routing_diagnostic.py      # Error-routing diagnostic plot
├── plot_and_test.py                      # Auxiliary-variance and return analysis
├── plot_salvage_return_boxplot.py        # Final-return reliability diagnostic
├── record_1_baseline.py                  # Render/evaluate Stage 1 checkpoint
├── record_2_surrogate.py                 # Render/evaluate Stage 2 checkpoint
├── record_3_paradox.py                   # Render/evaluate Stage 3 checkpoint
├── record_4_decoupling.py                # Render/evaluate Stage 4 checkpoint
├── 1_baseline.pth                        # Baseline PPO actor weights
├── 2_surrogate_hacking_attention.pth     # Differentiable-routing actor weights
├── 3_temporal_paradox_variance.pth       # Error-based-routing actor weights
├── 4_target_decoupling_final.pth         # Target Decoupling actor weights
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

Train or regenerate the differentiable-routing diagnostic:

```bash
python 2_surrogate_hacking_attention.py
python plot_surrogate_hacking_diagnostics.py
```

Generate the error-based routing diagnostic after the fixed five-seed TensorBoard runs are available locally:

```bash
python plot_error_routing_diagnostic.py
```

### Reproducibility: Error-Based Routing Runs

`plot_error_routing_diagnostic.py` uses a fixed set of five TensorBoard runs for the error-based routing diagnostic:

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
python 6_ablation_auxiliary_variance.py
python plot_and_test.py
python plot_salvage_return_boxplot.py
```

Evaluate rendered checkpoints with the pretrained actor weights in the repository root:

```bash
python record_1_baseline.py
python record_2_surrogate.py
python record_3_paradox.py
python record_4_decoupling.py
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
