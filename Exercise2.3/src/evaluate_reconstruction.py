"""
Quantitative evaluation: DDIM Inversion vs Null-Text Inversion (NTI).

For each test prompt we:
  1. Generate an image with DDIM sampling (this is our "original").
  2. Reconstruct it with DDIM inversion → sample.
  3. Reconstruct it with NTI → sample.
  4. Compute PSNR, SSIM, and LPIPS between original and each reconstruction.

Results are saved to:
  outputs/evaluation_results.csv   — per-image metrics
  outputs/evaluation_summary.png   — bar chart comparing mean metrics
  outputs/evaluation_grid.png      — visual grid: original / DDIM / NTI

Requirements:
    pip install scikit-image lpips
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms as tfms

# ── project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_setup import pipe, device, vae_scale_factor
from ddim_sampling import sample
from ddim_inversion import invert
from null_text_inversion import null_text_inversion

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── evaluation settings ──────────────────────────────────────────────────────
NUM_STEPS       = 50
GUIDANCE_SCALE  = 3.5
NTI_OPT_STEPS   = 10    # gradient steps per timestep in NTI (more = better)
NTI_LR          = 1e-2

# Each tuple: (generation_prompt, inversion_prompt)
# The inversion prompt is a shorter caption used during DDIM/NTI inversion.
TEST_PROMPTS = [
    ("a golden retriever running on a beach",   "A dog on the beach"),
    ("a red barn surrounded by a snowy field",  "A barn in snow"),
    ("a busy street market in Tokyo at night",  "A street market"),
    ("a lighthouse standing at sunset",         "A lighthouse"),
    ("a wooden bowl filled with fresh fruit",   "Fruit in a bowl"),
    ("a vintage bicycle leaning against a wall","A bicycle"),
    ("a waterfall in a tropical rainforest",    "A waterfall"),
    ("a cozy library with leather reading chairs", "A library interior"),
    ("a white sailboat on calm ocean water",    "A sailboat"),
    ("a mountain peak with clouds far below",   "A mountain"),
    ("a child playing in autumn leaves",        "A child outdoors"),
    ("a cat sleeping on a sunny windowsill",    "A sleeping cat"),
]


# ── metric helpers ───────────────────────────────────────────────────────────

def pil_to_np(img):
    """PIL Image → float32 numpy array in [0, 1], shape (H, W, 3)."""
    return np.array(img).astype(np.float32) / 255.0


def compute_psnr(orig_np, recon_np):
    from skimage.metrics import peak_signal_noise_ratio
    return peak_signal_noise_ratio(orig_np, recon_np, data_range=1.0)


def compute_ssim(orig_np, recon_np):
    from skimage.metrics import structural_similarity
    return structural_similarity(orig_np, recon_np, channel_axis=2, data_range=1.0)


def np_to_lpips_tensor(img_np):
    """float32 (H, W, 3) in [0,1] → LPIPS tensor (1, 3, H, W) in [-1, 1]."""
    t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return t * 2.0 - 1.0  # scale to [-1, 1] as expected by LPIPS


# ── main evaluation loop ─────────────────────────────────────────────────────

def encode_image(pil_img):
    """Encode a PIL image to a latent tensor."""
    with torch.no_grad():
        latent = pipe.vae.encode(
            tfms.functional.to_tensor(pil_img).unsqueeze(0).to(device) * 2 - 1
        )
    return vae_scale_factor * latent.latent_dist.sample()


def run_ddim_reconstruction(latent, prompt):
    """Invert latent → z_T, then reconstruct with DDIM sampling."""
    inverted = invert(latent, prompt, guidance_scale=GUIDANCE_SCALE,
                      num_inference_steps=NUM_STEPS)
    recon = sample(
        prompt,
        start_latents=inverted[-1][None],
        start_step=0,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    )
    return recon[0]


def run_nti_reconstruction(latent, prompt):
    """NTI: optimise null-text embeddings, then reconstruct."""
    z_T, null_texts = null_text_inversion(
        latent, prompt,
        guidance_scale=7.5,          # NTI needs higher CFG for quality
        num_inference_steps=NUM_STEPS,
        num_opt_steps=NTI_OPT_STEPS,
        lr=NTI_LR,
    )
    recon = sample(
        prompt,
        start_latents=z_T.to(device),
        guidance_scale=7.5,
        num_inference_steps=NUM_STEPS,
        null_texts=null_texts,
    )
    return recon[0]


def main():
    # ── load LPIPS ────────────────────────────────────────────────────────────
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(device)
    except ImportError:
        print("WARNING: lpips not installed. LPIPS will be skipped.")
        print("         Install with: uv pip install lpips")
        lpips_fn = None

    rows = []         # list of dicts for CSV
    originals, ddim_recons, nti_recons, titles = [], [], [], []

    for idx, (gen_prompt, inv_prompt) in enumerate(TEST_PROMPTS):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(TEST_PROMPTS)}] {gen_prompt}")
        print(f"{'='*60}")

        # 1. Generate "original" image
        print("  Generating original image...")
        orig_imgs = sample(gen_prompt, num_inference_steps=NUM_STEPS,
                           guidance_scale=GUIDANCE_SCALE)
        orig_pil = orig_imgs[0]
        orig_np  = pil_to_np(orig_pil)

        # Encode original to latent for inversion
        latent = encode_image(orig_pil)

        # 2. DDIM reconstruction
        print("  Running DDIM inversion + reconstruction...")
        ddim_pil = run_ddim_reconstruction(latent, inv_prompt)
        ddim_np  = pil_to_np(ddim_pil)

        # 3. NTI reconstruction
        print("  Running NTI reconstruction (slower)...")
        nti_pil = run_nti_reconstruction(latent, inv_prompt)
        nti_np  = pil_to_np(nti_pil)

        # 4. Compute metrics
        ddim_psnr = compute_psnr(orig_np, ddim_np)
        ddim_ssim = compute_ssim(orig_np, ddim_np)
        nti_psnr  = compute_psnr(orig_np, nti_np)
        nti_ssim  = compute_ssim(orig_np, nti_np)

        ddim_lpips_val = nti_lpips_val = float("nan")
        if lpips_fn is not None:
            with torch.no_grad():
                orig_t = np_to_lpips_tensor(orig_np).to(device)
                ddim_t = np_to_lpips_tensor(ddim_np).to(device)
                nti_t  = np_to_lpips_tensor(nti_np).to(device)
                ddim_lpips_val = lpips_fn(orig_t, ddim_t).item()
                nti_lpips_val  = lpips_fn(orig_t, nti_t).item()

        print(f"  DDIM → PSNR={ddim_psnr:.2f}  SSIM={ddim_ssim:.4f}  LPIPS={ddim_lpips_val:.4f}")
        print(f"  NTI  → PSNR={nti_psnr:.2f}  SSIM={nti_ssim:.4f}  LPIPS={nti_lpips_val:.4f}")

        rows.append({
            "prompt": gen_prompt,
            "ddim_psnr": ddim_psnr, "ddim_ssim": ddim_ssim, "ddim_lpips": ddim_lpips_val,
            "nti_psnr":  nti_psnr,  "nti_ssim":  nti_ssim,  "nti_lpips":  nti_lpips_val,
        })

        originals.append(orig_pil)
        ddim_recons.append(ddim_pil)
        nti_recons.append(nti_pil)
        titles.append(gen_prompt[:30])

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = OUTPUTS_DIR / "evaluation_results.csv"
    fieldnames = ["prompt",
                  "ddim_psnr", "ddim_ssim", "ddim_lpips",
                  "nti_psnr",  "nti_ssim",  "nti_lpips"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {csv_path}")

    # ── Summary statistics ────────────────────────────────────────────────────
    def mean(key): return np.mean([r[key] for r in rows])

    print("\n--- Average Metrics ---")
    print(f"{'Method':<8} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print(f"{'DDIM':<8} {mean('ddim_psnr'):>8.2f} {mean('ddim_ssim'):>8.4f} {mean('ddim_lpips'):>8.4f}")
    print(f"{'NTI':<8}  {mean('nti_psnr'):>8.2f}  {mean('nti_ssim'):>8.4f}  {mean('nti_lpips'):>8.4f}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = [
        ("PSNR ↑",  "ddim_psnr",  "nti_psnr"),
        ("SSIM ↑",  "ddim_ssim",  "nti_ssim"),
        ("LPIPS ↓", "ddim_lpips", "nti_lpips"),
    ]
    for ax, (label, ddim_key, nti_key) in zip(axes, metrics):
        vals = [mean(ddim_key), mean(nti_key)]
        bars = ax.bar(["DDIM", "NTI"], vals, color=["steelblue", "darkorange"])
        ax.set_title(label)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_ylim(0, max(vals) * 1.25)
    plt.suptitle("Reconstruction Quality: DDIM Inversion vs NTI")
    plt.tight_layout()
    summary_path = OUTPUTS_DIR / "evaluation_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {summary_path}")

    # ── Visual grid ───────────────────────────────────────────────────────────
    n = len(originals)
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 3))
    if n == 1:
        axes = axes[None]   # ensure 2-D indexing works
    col_titles = ["Original", "DDIM Recon", "NTI Recon"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")
    for row in range(n):
        for col, img in enumerate([originals[row], ddim_recons[row], nti_recons[row]]):
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(titles[row], fontsize=7, rotation=0,
                                labelpad=60, va="center")
    plt.tight_layout()
    grid_path = OUTPUTS_DIR / "evaluation_grid.png"
    plt.savefig(grid_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved {grid_path}")


if __name__ == "__main__":
    main()
