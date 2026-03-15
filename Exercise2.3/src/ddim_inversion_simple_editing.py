"""
Task: Image editing via DDIM inversion.

This script demonstrates the full DDIM-based editing pipeline:
  1. Encode the input image to latent space.
  2. Run DDIM inversion to obtain intermediate noisy latents.
  3. Start denoising from an intermediate step with a *new* prompt.

The key insight: by starting denoising from an intermediate inverted latent,
the model preserves the structure of the original image while applying the
new prompt.  The `start_step` parameter controls the trade-off:
  - Low start_step  → more faithful to original structure (less edit freedom)
  - High start_step → more aggressive editing (less structure preserved)

Try the script with your own group photo and edit prompt!
"""

import torch
from torchvision import transforms as tfms
from diffusers.utils import load_image
from PIL import Image

from pipeline_setup import pipe, device, vae_scale_factor
from ddim_sampling import sample
from ddim_inversion import invert


def edit(input_image, input_image_prompt, edit_prompt,
         num_steps=100, start_step=30, guidance_scale=3.5):
    """
    Edit an image by DDIM inversion + re-sampling with a new prompt.

    Parameters
    ----------
    input_image : PIL.Image
        The source image to edit (should be 512×512).
    input_image_prompt : str
        A text description of the *source* image (used during inversion).
    edit_prompt : str
        The target description for the edited image.
    num_steps : int
        Number of DDIM inversion/sampling steps. More steps → more accurate.
    start_step : int
        Which inverted latent to start sampling from.
        Larger values allow more structural change.
    guidance_scale : float
        CFG scale for the sampling pass.

    Returns
    -------
    PIL.Image
        The edited image.
    """
    # Encode the input image to latent space
    with torch.no_grad():
        latent = pipe.vae.encode(
            tfms.functional.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1
        )
    l = vae_scale_factor * latent.latent_dist.sample()

    # Run DDIM inversion to get the trajectory of noisy latents
    inverted_latents = invert(l, input_image_prompt, num_inference_steps=num_steps)

    # Sample (denoise) from the intermediate inverted latent with the new prompt
    final_im = sample(
        edit_prompt,
        start_latents=inverted_latents[-(start_step + 1)][None],
        start_step=start_step,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    )[0]

    return final_im


if __name__ == "__main__":
    import sys
    import yaml
    import matplotlib.pyplot as plt
    from pathlib import Path

    CONFIGS_DIR = Path(__file__).parent.parent / "configs"
    OUTPUTS_ROOT = Path(__file__).parent.parent / "outputs"

    # Optional: pass specific config filenames as CLI args, e.g.:
    #   uv run python ddim_inversion_simple_editing.py ddim_baseline.yaml
    # Default: run all ddim_*.yaml configs in alphabetical order.
    patterns = sys.argv[1:] or ["ddim_*.yaml"]

    config_files = []
    for pat in patterns:
        config_files.extend(sorted(CONFIGS_DIR.glob(pat)))

    if not config_files:
        print(f"No config files found matching {patterns} in {CONFIGS_DIR}")
        sys.exit(1)

    print(f"Found {len(config_files)} config(s) to run.")

    for cfg_path in config_files:
        cfg = yaml.safe_load(cfg_path.read_text())

        run_dir = OUTPUTS_ROOT / cfg["run_name"]
        run_dir.mkdir(parents=True, exist_ok=True)

        img_path = (cfg_path.parent / cfg["input_image"]).resolve()
        input_image = Image.open(img_path).convert("RGB").resize((512, 512))

        print(f"\n[{cfg['run_name']}]  "
              f"num_steps={cfg['num_steps']}  "
              f"start_step={cfg['start_step']}  "
              f"guidance_scale={cfg['guidance_scale']}")
        print(f"  prompt: \"{cfg['edit_prompt']}\"")

        result = edit(
            input_image,
            input_image_prompt=cfg["input_image_prompt"],
            edit_prompt=cfg["edit_prompt"],
            num_steps=cfg["num_steps"],
            start_step=cfg["start_step"],
            guidance_scale=cfg["guidance_scale"],
        )

        input_image.save(run_dir / "original.png")
        result.save(run_dir / "edited.png")

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(input_image)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(result)
        axes[1].set_title(
            f"{cfg['edit_prompt'][:45]}\n"
            f"steps={cfg['num_steps']}  start={cfg['start_step']}  cfg={cfg['guidance_scale']}"
        )
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(run_dir / "comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  → Saved to {run_dir}/")
