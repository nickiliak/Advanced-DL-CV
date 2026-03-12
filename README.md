# Advanced Deep Learning in Computer Vision

Advanced Deep Learning in Computer Vision course exercises and assignments from DTU (02501).

## Course Info

- **Course Code**: 02501
- **Course Name**: Advanced Deep Learning in Computer Vision
- **University**: DTU (Technical University of Denmark)

## Repository Structure

```
Advanced-DL-CV/
├── Exercise1.1/          # Transformers & Einops
├── Exercise1.2/          # (Week 1)
├── Exercise1.3/          # (Week 1)
├── Exercise2.1/          # Denoising Diffusion Probabilistic Models (DDPM)
├── Exercise2.2/          # Conditional Diffusion & Classifier-Free Guidance
├── Exercise2.3/          # (Week 2)
├── Hand-ins/             # Submitted assignments
├── outputs/              # Shared outputs
├── pyproject.toml        # Project configuration and dependencies
└── uv.lock               # Dependency lock file
```

### Exercise 1.1 - Transformers & Einops

Introduction to transformers and the einops library. Includes a transformer implementation, text classification, and einops tutorials.

### Exercise 2.1 - DDPM

Implementation of Denoising Diffusion Probabilistic Models. Covers the forward/reverse diffusion process, UNet training, and sample generation on a sprites dataset. Configured via YAML (`configs/exercise2.1.yaml`).

### Exercise 2.2 - Conditional Diffusion & Classifier-Free Guidance

Extends DDPM with classifier guidance and classifier-free guidance. Includes classifier training/evaluation, FID score computation, and TensorBoard logging.

## Setup

Requires Python >= 3.11. Install dependencies using `uv`:

```bash
uv sync
```

Run scripts:

```bash
uv run python <script_path>
```

## Project Layout

Each exercise directory follows a common structure:

- `src/` - Source code and scripts
- `configs/` - Configuration files (YAML)
- `models/` - Model checkpoints
- `outputs/` - Generated outputs and results
- `data/` - Datasets
- `week_report/` - Weekly reports
