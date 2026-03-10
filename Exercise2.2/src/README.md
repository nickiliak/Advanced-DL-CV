
# Exercise 2.2 - Running the Code

## classifier_train.py

Trains a classifier for diffusion models. Logs training loss to Tensorboard.

### Default run
```bash
uv run python Exercise2.2/src/classifier_train.py
```

**Available Parameters (via train()):**
- `--device`: Device to run on (default: 'cpu')
- `--T`: Diffusion steps (default: 500)
- `--img_size`: Image size (default: 16)
- `--input_channels`: Input channels (default: 3)
- `--channels`: Model channels (default: 32)
- `--time_dim`: Time embedding dim (default: 256)

---

## classifier_eval.py

Evaluates classifier accuracy per timestep and visualizes qualitative results.

### Default run
```bash
uv run python Exercise2.2/src/classifier_eval.py
```

---

## ddpm_train.py

Trains a diffusion model. Supports classifier-free guidance with `--cfg` flag. Logs loss to Tensorboard.

### Default run
```bash
uv run python Exercise2.2/src/ddpm_train.py
```

### With classifier-free guidance
```bash
uv run python Exercise2.2/src/ddpm_train.py --cfg
```

**Available Parameters (via train()):**
- `--T`: Diffusion steps (default: 500)
- `--cfg`: Classifier-free guidance (default: True)
- `--img_size`: Image size (default: 16)
- `--input_channels`: Input channels (default: 3)
- `--channels`: Model channels (default: 32)
- `--time_dim`: Time embedding dim (default: 256)
- `--batch_size`: Batch size (default: 100)
- `--lr`: Learning rate (default: 1e-3)
- `--num_epochs`: Number of epochs (default: 30)
- `--experiment_name`: Experiment name (default: 'DDPM-cfg')
- `--show`: Show images (default: False)
- `--device`: Device to run on (default: 'cpu')

---

## ddpm_eval.py

Evaluates diffusion models and computes FID scores for classifier guidance and classifier-free guidance.

### Default run
```bash
uv run python Exercise2.2/src/ddpm_eval.py
```

---

## Logging
* The code uses tensorboard to log the train loss. Use the command:
	```bash
	tensorboard --logdir=runs
	```
	to observe the training loss.