# Exercise 2.3: Diffusion-Based Image Editing

This directory contains implementations of diffusion-based image editing techniques, including DDIM sampling, inversion, Null-Text Inversion, and cross-attention control.

## Overview

The scripts build on each other:
1. **ddim_sampling.py** - Basic DDIM denoising to generate images from text
2. **ddim_inversion.py** - Invert images to noise, then reconstruct
3. **null_text_inversion.py** - Optimize null-text embeddings for perfect reconstruction
4. **vis_cross_attn.py** - Visualize which pixels attend to which words
5. **image_to_image_simple_editing.py** - Simple img2img editing (add noise + denoise)
6. **ddim_inversion_simple_editing.py** - Structure-preserving editing via inversion
7. **cross_attention_editing.py** - Advanced editing with cross-attention control

---

## Running the Scripts

### 1. DDIM Sampling
Generate an image from a text prompt using DDIM denoising.

```bash
uv run python Exercise2.3/src/ddim_sampling.py
```

**Output:** `ddim_sample_output.png`

**Default parameters:**
- `prompt`: "Watercolor painting of Nyhavn, Copenhagen"
- `negative_prompt`: "blurry, ugly, stock photo"
- `num_inference_steps`: 50
- `guidance_scale`: 3.5

---

### 2. DDIM Inversion
Invert a real image to noise, then reconstruct it perfectly via DDIM sampling.

```bash
uv run python Exercise2.3/src/ddim_inversion.py
```

**Outputs:**
- `ddim_inverted_noisy.png` - The noisy latent decoded to pixel space
- `ddim_reconstruction.png` - Original and reconstructed side-by-side

**Default parameters:**
- `NUM_STEPS`: 50 (number of inversion/sampling steps)
- `START_STEP`: 0 (start from pure noise for strictest reconstruction test)
- Input image: Puppy on grass (from Pexels)

---

### 3. Null-Text Inversion (NTI)
Optimize per-timestep null-text embeddings for near-perfect image reconstruction.

```bash
uv run python Exercise2.3/src/null_text_inversion.py
```

**Outputs:**
- `nti_reconstruction.png` - Reconstructed image
- `nti_original.png` - Original image

**Default parameters:**
- `NUM_STEPS`: 50
- `guidance_scale`: 7.5
- `num_opt_steps`: 10 (gradient steps per timestep)
- `lr`: 1e-2 (learning rate for optimization)
- Input image: Puppy on grass (from Pexels)

---

### 4. Cross-Attention Visualization
Generate an image and visualize which pixels each word in the prompt attends to.

```bash
uv run python Exercise2.3/src/vis_cross_attn.py
```

**Output:** `cross_attn_vis.png`

**Default parameters:**
- `prompt`: "a fluffy cat sitting on a wooden table"
- `num_inference_steps`: 20
- `guidance_scale`: 7.5
- `res`: 16 (spatial resolution of attention maps: 16×16)

---

### 5. Image-to-Image Simple Editing
Simple img2img approach: add noise to the image, then denoise with a new prompt.

```bash
uv run python Exercise2.3/src/image_to_image_simple_editing.py
```

**Output:** `img2img_edit_comparison.png` (original + edits at different noise levels)

**Default parameters:**
- `input_image`: Puppy on grass (from Pexels)
- `edit_prompt`: "Photograph of a cat on the grass"
- `num_steps`: 50
- `start_steps`: [5, 10, 20, 30] (noise levels to try)
- `guidance_scale`: 3.5

**Effect of `start_step`:**
- Low (5-10): Subtle edits, background mostly preserved
- High (20-30): Aggressive edits, more structural changes

---

### 6. DDIM Inversion-Based Editing
Structure-preserving editing: invert the source image, then re-sample with a new prompt.

```bash
uv run python Exercise2.3/src/ddim_inversion_simple_editing.py
```

**Outputs:**
- `edit_puppy_to_cat.png` - Edited image (puppy → cat)
- `edit_original.png` - Original image
- (Commented-out example: `edit_group_sunglasses.png` with your own image)

**Default parameters:**
- `input_image`: Puppy on grass (from Pexels)
- `input_image_prompt`: "A puppy on the grass"
- `edit_prompt`: "A cat on the grass"
- `num_steps`: 50
- `start_step`: 10 (lower = more structure preserved)
- `guidance_scale`: 3.5

**Hyperparameters to explore:**
- `num_steps`: 50–500 (more → more accurate but slower)
- `start_step`: 5–50 (higher → bigger edit, less structure)
- `guidance_scale`: 3–10 (higher → stronger prompt adherence)

---

### 7. Cross-Attention Control Editing
Advanced editing with cross-attention injection for precise structure preservation.

```bash
uv run python Exercise2.3/src/cross_attention_editing.py
```

**Output:** `cross_attn_edit_result.png` (original, reconstructed, edited side-by-side)

**Default parameters:**
- `input_image`: Puppy on grass (from Pexels)
- `source_prompt`: "A photograph of a puppy on the grass"
- `target_prompt`: "A photograph of a cat on the grass"
- `num_inference_steps`: 50
- `guidance_scale`: 7.5
- `injection_threshold`: 0.5 (inject attention maps for first 50% of steps)
- `use_nti`: True (use Null-Text Inversion for better reconstruction)

**Parameters:**
- `injection_threshold`: Controls where to inject source attention maps
  - 0.0: No injection (free generation)
  - 0.5: Inject for first 50% of steps (good balance)
  - 1.0: Always inject (strict structure preservation)
- `use_nti`:
  - True: Uses optimized null-text embeddings (slower but better quality)
  - False: Uses plain DDIM inversion (faster)

---

## Comparing Editing Techniques

| Technique | Speed | Structure Preservation | Edit Freedom |
|-----------|-------|----------------------|--------------|
| img2img | Fast | Low | High |
| DDIM Inversion | Moderate | Moderate | Moderate |
| DDIM + Cross-Attn | Moderate | High | Moderate |
| NTI + Cross-Attn | Slow | Very High | Low |

---

## Common Issues

**Issue:** Image download fails
- **Solution:** The scripts fetch images from external URLs. If you're offline or the URLs are inaccessible, replace with local image paths.

**Issue:** CUDA out of memory
- **Solution:** Reduce `num_inference_steps` or use a smaller model (e.g., `sd-turbo`).

**Issue:** Poor reconstruction quality
- **Solution:** Increase `num_steps` in inversion (try 100+) or increase NTI `num_opt_steps`.

---

## References

- **DDIM:** [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- **Prompt-to-Prompt (Cross-Attention Control):** [Hertz et al., 2022](https://arxiv.org/abs/2208.01626)
- **Null-Text Inversion:** [Mokady et al., 2023](https://arxiv.org/abs/2211.09794)
