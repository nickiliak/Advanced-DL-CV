# Claude.md: Cross-Attention Visualization Teaching Guide

## Teaching Approach: Socratic Scaffolding
This document uses the **Socratic method** to guide your learning of cross-attention visualization in Stable Diffusion. Rather than giving you the full solution, I'll:
- Ask what you already understand
- Provide conceptual analogies, technical hints, and probing questions
- Guide you through a learning roadmap
- Encourage you to write and explain code incrementally

---

## Learning Roadmap: Understanding Cross-Attention

### Phase 1: Conceptual Foundation
**Goal:** Understand what attention maps are and why we aggregate them.

**Baseline Question:** Before we dive in, tell me:
1. What do you understand by "attention weights" in the context of neural networks?
2. Why might we want to average attention across multiple layers and heads?

**What we're building toward:** A visualization showing which pixels each word in your prompt influences during image generation.

---

### Phase 2: Shape Transformations (The Core Challenge)

This is where most of the work happens. The hard part isn't the computation—it's tracking tensor shapes.

**Conceptual Analogy:**
Think of reshaping tensors like reorganizing a filing cabinet:
- You have papers stacked in one configuration: `(batch*heads, spatial, tokens)`
- You need to reorganize them into cabinets (batches) with folders (heads) with documents (tokens): `(batch, heads, spatial, tokens)`
- Then you toss out duplicate info (average the heads) and reshape the spatial location

**Technical Hint:**
When you see `(batch * heads, H*W, seq)`, remember:
- `batch * heads` is the **product** of two dimensions that need to be **separated**
- `H*W` is a **flattened** 2D grid that needs to be **reshaped** back to `(H, W)` or `(res, res)`
- `seq` is the number of text tokens—this one typically stays as-is

**Your Turn (Probing Question):**
Given a tensor of shape `(32, 256, 77)` where:
- `32 = 4 batches × 8 heads`
- `256 = 16 × 16 (spatial dimension)`
- `77 = number of text tokens`

What would the shape be after:
1. Separating batch and heads? ➜ `(?, ?, ?, ?)`
2. Averaging over heads? ➜ `(?, ?, ?)`
3. Reshaping 256 back to 16×16? ➜ `(?, ?, ?, ?)`

---

### Phase 3: Filtering by Resolution

**The Question:** Why does the code only keep layers where `attention_maps[i].shape[1] == res*res`?

**Technical Hint:**
Different layers in the UNet operate at different spatial resolutions (8×8, 16×16, 32×32, 64×64). If you want a 16×16 visualization, you only care about layers that *already have* 16×16 resolution. Using mismatched resolutions would require interpolation and waste information.

**Your Turn:**
In the function `aggregate_attention_maps(attention_maps, num_heads, res=16)`:
- How would you *filter* the list `attention_maps` to keep only those with shape `(batch*heads, res*res, seq_len)`?
- Hint: list comprehensions + `.shape` attribute

---

### Phase 4: Stacking and Final Averaging

**Conceptual Analogy:**
Imagine you have multiple photographs of the same scene taken with different cameras (layers) and different lenses (heads). You:
1. Normalize each photo's exposure
2. Average them all together into a single image
3. That's your "consensus attention map"

**Technical Hint:**
PyTorch's `torch.stack()` combines tensors along a *new* dimension, while `.mean()` averages along an *existing* dimension.

**Your Turn:**
If you have a list of 4 tensors, each of shape `(1, 77, 16, 16)`, what's:
1. The shape after `torch.stack()`? ➜ `(?, ?, ?, ?, ?)`
2. After averaging on dimension 0? ➜ `(?, ?, ?, ?)`

---

## Implementation Checklist

Once you've worked through the phases above, here's what your function needs to do in order:

- [ ] **Filter:** Keep only maps where spatial dim = `res*res`
- [ ] **Loop through filtered maps:**
  - [ ] Reshape `(batch*heads, H*W, seq)` → `(batch, heads, H*W, seq)`
  - [ ] Average heads → `(batch, H*W, seq)`
  - [ ] Reshape H*W → `(batch, seq, res, res)`
- [ ] **Stack:** Combine all per-layer maps into a single tensor
- [ ] **Average:** Take mean across all layers
- [ ] **Squeeze:** Remove batch dimension → `(seq, res, res)`

---

## Resources to Reference

- **Tensor shape operations:** Look at how `attn.head_to_batch_dim()` and `attn.batch_to_head_dim()` work in the `AttentionCapture` class (they do the opposite of what you need!)
- **PyTorch basics:** `torch.stack()`, `.reshape()`, `.mean()`, list comprehensions
- **Understanding the flow:** Trace through `visualize_cross_attention()` to see where your function fits in the pipeline

---

## When You Get Stuck

1. **Print shapes:** Add `print(tensor.shape)` at each step to verify your reshapes
2. **Ask a probing question:** "What dimension am I collapsing?" or "What new dimension do I need?"
3. **Reference the existing code:** How does `AttentionCapture.__call__()` manipulate shapes? Can you learn from it?

---

## Success Criteria

You'll know you're done when:
- `aggregate_attention_maps()` returns a tensor of shape `(num_text_tokens, res, res)` ✓
- The visualization shows sensible heatmaps (bright pixels = words attend there) ✓
- You can explain *why* each reshape is necessary, not just *how* to do it ✓
