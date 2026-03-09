# Exercise 2.1 - Running the Code

All scripts use a YAML config file for parameters: `Exercise2.1/configs/exercise2.1.yaml`.

## playground.py

Visualizes the diffusion process and saves example images to the assets directory.

### Default run
```bash
uv run python Exercise2.1/src/playground.py
```

### With custom config
Edit `Exercise2.1/configs/exercise2.1.yaml` to change:
- `seed`: Random seed
- `batch_size`: Number of images to visualize
- `assets_dir`: Output directory for images
- `sprites_data`: Path to sprites.npy

**Available Parameters (via config)**:
- `seed` (default: 2929)
- `batch_size` (default: 8)
- `assets_dir` (default: ../assets)
- `sprites_data` (default: ../data/sprites.npy)
- `num_samples` (default: 40000)

---

## ddpm_train.py

Trains a DDPM model on the sprites dataset. Saves checkpoints and generated samples.

### Default run
```bash
uv run python Exercise2.1/src/ddpm_train.py
```

### With custom config
Edit `Exercise2.1/configs/exercise2.1.yaml` to change:
- `T`: Total diffusion steps
- `img_size`: Image size
- `input_channels`: Input channels
- `channels`: Model channels
- `time_dim`: Time embedding dimension
- `batch_size`: Training batch size
- `lr`: Learning rate
- `num_epochs`: Number of epochs
- `experiment_name`: Experiment folder name
- `models_dir`, `outputs_dir`: Output directories

**Available Parameters (via config)**:
- `seed` (default: 2929)
- `batch_size` (default: 8)
- `num_samples` (default: 40000)
- `img_size` (default: 16)
- `T` (default: 500)
- `beta_start` (default: 0.0001)
- `beta_end` (default: 0.02)
- `input_channels` (default: 3)
- `channels` (default: 32)
- `time_dim` (default: 256)
- `lr` (default: 0.001)
- `num_epochs` (default: 30)
- `experiment_name` (default: ddpm)
- `models_dir` (default: ../models)
- `outputs_dir` (default: ../outputs)
- `show_images` (default: true)

---

**To run with custom parameters, edit the YAML config and rerun the script.**

**Customization**:
To modify training parameters, edit the `train()` function call in the `main()` function or add command-line argument support (argparse) to the script.

**Output**:
- Model checkpoints: `models/<experiment_name>/weights-<epoch>.pt`
- Generated samples: `results/<experiment_name>/<epoch>.jpg`
- TensorBoard logs: `runs/<experiment_name>/`
