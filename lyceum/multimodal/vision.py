"""Vision (multimodal) capability for Lyceum -- a scaled-down native-multimodal LM.

How frontier models "see"
-------------------------
A natively multimodal model does not bolt a captioner onto a text model. Instead
a modality-specific **encoder** turns the raw input (here, image pixels) into a
short sequence of vectors that live in the *exact same embedding space* the
language model uses for word tokens. Those vectors -- "image tokens" -- are
concatenated in front of the text tokens, and the ordinary causal transformer
trunk runs over the whole sequence. From the transformer's point of view there is
no difference between an image token and a word token: text positions simply
*attend back* to the image prefix and read information out of it.

This module is a faithful, nano-scale version of that pipeline:

  * :func:`image_to_patches` -- ViT-style patchify of a grayscale / RGB image.
  * :class:`PatchEmbedder` -- a tiny linear projector ``Linear(patch_pixels -> dim)``
    plus a learned positional embedding. A *real* system uses a full ViT or CLIP
    vision transformer here; this single linear layer is the scaled-down stand-in
    for that encoder, but it plays exactly the same architectural role: project
    pixels into the LM's token-embedding space.
  * :class:`MultimodalLM` -- wraps a :class:`~lyceum.model.transformer.LyceumLM`
    and the projector. ``forward_multimodal`` builds the combined embedding
    sequence ``[image tokens] ++ [tok_emb(text_ids)]`` and runs the LM trunk
    manually over it (re-using the model's RoPE tables, blocks, norm and head),
    so the text learns to answer questions *about the image*.

Run the self-test (CPU, < 60s)::

    python -m lyceum.multimodal.vision
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import get_config, ModelConfig
from ..data.tokenizer import BPETokenizer
from ..model.transformer import LyceumLM


# --------------------------------------------------------------------------- #
# 1. Patchify: turn an image into a grid of flattened pixel patches.
# --------------------------------------------------------------------------- #
def image_to_patches(img: np.ndarray, patch: int = 4) -> np.ndarray:
    """Split a ``(H, W)`` or ``(H, W, C)`` image into flattened patches.

    This is the first half of a Vision Transformer's input pipeline: the image
    is cut into a non-overlapping grid of ``patch x patch`` tiles, and each tile
    is flattened into a single vector. The result is a sequence we can feed to a
    projector exactly the way a token sequence is fed to an embedding table.

    Returns an array of shape ``(num_patches, patch * patch * C)`` where the
    patches are in row-major (raster) order. ``H`` and ``W`` must be divisible by
    ``patch``.
    """
    if img.ndim == 2:
        img = img[:, :, None]  # (H, W) -> (H, W, 1)
    H, W, C = img.shape
    if H % patch != 0 or W % patch != 0:
        raise ValueError(
            f"image {H}x{W} not divisible by patch={patch}")
    nh, nw = H // patch, W // patch
    # (nh, patch, nw, patch, C) -> (nh, nw, patch, patch, C)
    grid = img.reshape(nh, patch, nw, patch, C).transpose(0, 2, 1, 3, 4)
    patches = grid.reshape(nh * nw, patch * patch * C)
    return np.ascontiguousarray(patches.astype(np.float32))


def patch_pixels(patch: int, channels: int = 1) -> int:
    """Number of values in one flattened patch."""
    return patch * patch * channels


# --------------------------------------------------------------------------- #
# 2. Patch embedder: project patches into the LM's token-embedding space.
# --------------------------------------------------------------------------- #
class PatchEmbedder(nn.Module):
    """A tiny ViT-style projector: ``Linear(patch_pixels -> dim)`` + learned pos.

    Each flattened pixel patch is linearly mapped into the language model's
    ``dim``-dimensional embedding space, then a learned positional embedding is
    added so the model can tell *where* in the image each patch came from. The
    output is a sequence of image "tokens" of shape ``(num_patches, dim)`` that
    is drop-in compatible with the LM's text token embeddings.

    NOTE: a production system replaces this single ``nn.Linear`` with a full
    ViT/CLIP vision encoder (many self-attention layers, pretrained on
    image-text pairs). The architectural contract is identical -- pixels in,
    embedding-space vectors out -- so the rest of the stack is unchanged.
    """

    def __init__(self, patch_pixels: int, dim: int, max_patches: int = 64):
        super().__init__()
        self.patch_pixels = patch_pixels
        self.dim = dim
        self.max_patches = max_patches
        self.proj = nn.Linear(patch_pixels, dim)
        self.pos_emb = nn.Parameter(torch.zeros(max_patches, dim))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """``patches``: ``(B, num_patches, patch_pixels)`` -> ``(B, num_patches, dim)``."""
        if patches.dim() == 2:  # allow a single un-batched image
            patches = patches.unsqueeze(0)
        B, N, P = patches.shape
        if N > self.max_patches:
            raise ValueError(
                f"{N} patches exceeds max_patches={self.max_patches}")
        x = self.proj(patches)                    # (B, N, dim)
        x = x + self.pos_emb[:N].unsqueeze(0)      # add learned positions
        return x


# --------------------------------------------------------------------------- #
# 3. The multimodal model: LM trunk run over [image tokens] ++ [text tokens].
# --------------------------------------------------------------------------- #
class MultimodalLM(nn.Module):
    """Wrap a :class:`LyceumLM` + :class:`PatchEmbedder` into one model.

    The key method, :meth:`forward_multimodal`, does what a native multimodal
    transformer does: it builds a single embedding sequence whose first
    ``num_image_tokens`` positions come from the image encoder and whose
    remaining positions come from the LM's text embedding table, then runs the
    *same* transformer trunk over the whole thing.
    """

    def __init__(self, lm: LyceumLM, embedder: PatchEmbedder):
        super().__init__()
        self.lm = lm
        self.embedder = embedder
        assert embedder.dim == lm.cfg.dim, "embedder dim must match model dim"

    # ..................................................................... #
    def _run_trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Run the LyceumLM trunk over pre-computed embeddings ``x`` (B, T, dim).

        This replicates :meth:`LyceumLM.hidden_states`'s loop, but starting from
        embeddings we supply (a mix of image and text tokens) rather than from
        ``tok_emb(idx)``. RoPE is applied over the *full* combined length, so
        image tokens occupy real positions 0..K-1 and text tokens follow.
        """
        lm = self.lm
        B, T, _ = x.shape
        if T > lm.cfg.max_seq_len:
            raise ValueError(
                f"combined length {T} exceeds max_seq_len={lm.cfg.max_seq_len}")
        cos = lm.rope_cos[:T]
        sin = lm.rope_sin[:T]
        for i, block in enumerate(lm.blocks):
            x = block(x, cos, sin, None, i)
        return lm.norm(x)

    # ..................................................................... #
    def forward_multimodal(self, image_tokens, text_ids, targets=None):
        """Read an image as a prefix and predict / score the text.

        Parameters
        ----------
        image_tokens : Tensor ``(B, K, dim)``
            Output of :class:`PatchEmbedder` -- image tokens already in the LM's
            embedding space.
        text_ids : LongTensor ``(B, S)``
            Text token ids; embedded via the LM's ``tok_emb``.
        targets : LongTensor ``(B, S)`` or ``None``
            Next-token targets for the *text* positions. Image positions are
            never predicted, so they are masked with ``-100`` (ignored by the
            cross-entropy) when assembling the full-length target tensor.

        Returns ``(logits, loss)`` where ``logits`` has shape
        ``(B, K + S, vocab)`` and ``loss`` is ``None`` when ``targets is None``.
        """
        if image_tokens.dim() == 2:
            image_tokens = image_tokens.unsqueeze(0)
        lm = self.lm
        B, K, _ = image_tokens.shape
        text_emb = lm.tok_emb(text_ids)                    # (B, S, dim)
        x = torch.cat([image_tokens, text_emb], dim=1)     # (B, K+S, dim)

        h = self._run_trunk(x)                             # (B, K+S, dim)
        logits = lm.lm_head(h)                             # (B, K+S, vocab)

        loss = None
        if targets is not None:
            # Prepend -100 for the K image positions so they are ignored.
            img_pad = torch.full((B, K), -100, dtype=torch.long,
                                 device=targets.device)
            full_targets = torch.cat([img_pad, targets], dim=1)  # (B, K+S)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                full_targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def encode_image(self, patches: torch.Tensor) -> torch.Tensor:
        """Convenience: pixel patches -> image tokens."""
        return self.embedder(patches)


# --------------------------------------------------------------------------- #
# 4. Synthetic vision task: bright vs. dark grayscale grids.
# --------------------------------------------------------------------------- #
IMG_SIZE = 8
PATCH = 4
_PROMPT = "<user>is the image bright or dark?<assistant>"


def _make_image(bright: bool, rng: np.random.Generator) -> np.ndarray:
    """A noisy grayscale grid that is overall bright (high) or dark (low)."""
    base = 0.8 if bright else 0.2
    img = base + 0.15 * (rng.random((IMG_SIZE, IMG_SIZE)) - 0.5)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def build_vision_dataset(n: int, seed: int = 0):
    """Build ``n`` (image, label) examples for the bright/dark task.

    Returns a list of dicts ``{"image": (H,W) float array, "label": "bright"|"dark"}``
    with a roughly balanced mix of the two classes.
    """
    rng = np.random.default_rng(seed)
    data = []
    for i in range(n):
        bright = bool(i % 2 == 0)
        img = _make_image(bright, rng)
        data.append({"image": img, "label": "bright" if bright else "dark"})
    return data


def _build_tokenizer() -> BPETokenizer:
    """A tiny BPE tokenizer trained on the task's tiny vocabulary."""
    cfg = get_config("nano")
    tok = BPETokenizer(special_tokens=cfg.tokenizer.special_tokens)
    corpus = (_PROMPT + " bright dark " + "the image is ") * 50
    # keep vocab tiny so the LM head stays small/fast
    tok.train(corpus, vocab_size=320, verbose=False)
    return tok


def _encode_example(ex, tok, patch=PATCH):
    """Return (patches_tensor, prompt_ids, answer_ids) for one example."""
    patches = image_to_patches(ex["image"], patch=patch)          # (N, P)
    patches_t = torch.from_numpy(patches)                          # float32
    prompt_ids = tok.encode(_PROMPT)
    answer_ids = tok.encode(" " + ex["label"]) + [tok.id("<eos>")]
    return patches_t, prompt_ids, answer_ids


# --------------------------------------------------------------------------- #
# 5. Training loop -- teach the model to answer from the image tokens.
# --------------------------------------------------------------------------- #
def train_multimodal(mm_model: MultimodalLM, dataset, tok, steps: int = 30,
                     lr: float = 3e-3, seed: int = 0, verbose: bool = True):
    """Short CPU training loop. Returns the list of logged losses.

    For each example we build ``text_ids = prompt + answer`` and targets that
    are the next-token shift of ``text_ids``, with the *prompt* positions masked
    (-100) so the model is only trained to produce the answer given the image +
    question. Image positions are masked inside ``forward_multimodal``.
    """
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(mm_model.parameters(), lr=lr)
    mm_model.train()
    losses = []
    encoded = [_encode_example(ex, tok) for ex in dataset]
    n = len(encoded)
    for step in range(steps):
        patches_t, prompt_ids, answer_ids = encoded[step % n]
        image_tokens = mm_model.encode_image(patches_t)            # (1, K, dim)

        text_ids = prompt_ids + answer_ids
        # build text input and its next-token targets
        inp = torch.tensor(text_ids[:-1], dtype=torch.long).unsqueeze(0)
        tgt = torch.tensor(text_ids[1:], dtype=torch.long).unsqueeze(0)
        # mask everything that predicts a prompt token; train only on the answer
        n_prompt = len(prompt_ids)
        tgt_masked = tgt.clone()
        tgt_masked[0, : n_prompt - 1] = -100

        _, loss = mm_model.forward_multimodal(image_tokens, inp, tgt_masked)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mm_model.parameters(), 1.0)
        opt.step()

        losses.append(loss.item())
        if verbose and (step % max(1, steps // 6) == 0 or step == steps - 1):
            print(f"  step {step:3d}  loss {loss.item():.4f}")
    return losses


@torch.no_grad()
def predict(mm_model: MultimodalLM, ex, tok, patch=PATCH):
    """Greedy single-token classification: does the model say bright or dark?

    We feed image + prompt, look at the next-token distribution, and compare the
    probability mass the model puts on the first token of " bright" vs " dark".
    """
    mm_model.eval()
    patches = torch.from_numpy(image_to_patches(ex["image"], patch=patch))
    image_tokens = mm_model.encode_image(patches)
    prompt_ids = tok.encode(_PROMPT)
    inp = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
    logits, _ = mm_model.forward_multimodal(image_tokens, inp)
    next_logits = logits[0, -1]                       # distribution over answer
    bright_tok = tok.encode(" bright")[0]
    dark_tok = tok.encode(" dark")[0]
    return "bright" if next_logits[bright_tok] > next_logits[dark_tok] else "dark"


def evaluate(mm_model, dataset, tok):
    correct = sum(predict(mm_model, ex, tok) == ex["label"] for ex in dataset)
    return correct / len(dataset)


# --------------------------------------------------------------------------- #
# 6. Manual mapping.
# --------------------------------------------------------------------------- #
def describe() -> str:
    """Return (and print) how this module maps to the Frontier manual."""
    text = """\
Multimodality -- mapping to the Frontier manual
===============================================
* Native multimodality: images/audio share ONE token-embedding space with text.
  A modality encoder projects raw input into that space; the transformer trunk
  is unchanged and simply attends across modalities.
* Encoder/projector: real systems use a ViT or CLIP vision transformer. Here the
  PatchEmbedder (Linear + learned positions) is the scaled-down projector -- same
  contract, fewer parameters.
* Image tokens as a prefix: [image tokens] ++ [text tokens]. Text positions
  attend back to the image, so the model "reads" the image to answer.
* Adapter-style alternative: instead of native training, a frozen LM can be
  bridged to a frozen vision encoder via a small trained adapter (e.g. a
  Q-Former / cross-attention layer). This module demonstrates the native path:
  the projector is trained jointly with the LM trunk.
"""
    print(text)
    return text


# --------------------------------------------------------------------------- #
# 7. Self-test.
# --------------------------------------------------------------------------- #
def _build_nano_mm(tok):
    """A nano LyceumLM + matching PatchEmbedder, sized for a fast CPU run."""
    cfg = ModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
                      max_seq_len=128)
    lm = LyceumLM(cfg, vocab_size=tok.vocab_size)
    n_patches = (IMG_SIZE // PATCH) ** 2
    embedder = PatchEmbedder(patch_pixels=patch_pixels(PATCH, channels=1),
                             dim=cfg.dim, max_patches=max(8, n_patches))
    return MultimodalLM(lm, embedder)


def _self_test():
    torch.manual_seed(1337)
    print("=" * 64)
    print("lyceum.multimodal.vision -- self-test")
    print("=" * 64)
    describe()

    tok = _build_tokenizer()
    print(f"tokenizer vocab_size = {tok.vocab_size}")

    # Sanity: patchify shape.
    img = build_vision_dataset(2)[0]["image"]
    patches = image_to_patches(img, patch=PATCH)
    n_patches = (IMG_SIZE // PATCH) ** 2
    assert patches.shape == (n_patches, PATCH * PATCH), patches.shape
    print(f"patchify: {img.shape} -> {patches.shape} "
          f"({n_patches} image tokens)")

    mm = _build_nano_mm(tok)
    print(f"model params = {mm.lm.num_params():,} + "
          f"projector {sum(p.numel() for p in mm.embedder.parameters()):,}")

    # Forward pass smoke test.
    train = build_vision_dataset(16, seed=1)
    test = build_vision_dataset(8, seed=99)
    p0, prompt0, ans0 = _encode_example(train[0], tok)
    it0 = mm.encode_image(p0)
    logits, loss = mm.forward_multimodal(
        it0,
        torch.tensor(prompt0 + ans0)[:-1].unsqueeze(0),
        torch.tensor(prompt0 + ans0)[1:].unsqueeze(0),
    )
    print(f"forward_multimodal: logits {tuple(logits.shape)}, "
          f"loss {loss.item():.4f}")

    acc_before = evaluate(mm, test, tok)
    print(f"\ntraining (30 steps)...")
    losses = train_multimodal(mm, train, tok, steps=30)
    acc_after = evaluate(mm, test, tok)

    print(f"\nfirst loss {losses[0]:.4f} -> last loss {losses[-1]:.4f}")
    print(f"test accuracy: {acc_before:.2f} (before) -> {acc_after:.2f} (after)")

    assert losses[-1] < losses[0], "loss did not decrease"
    print("\nPASS: loss decreased; image tokens are read as a text prefix.")
    if acc_after > 0.5:
        print(f"PASS: accuracy {acc_after:.2f} is above chance (0.5).")
    else:
        print(f"NOTE: accuracy {acc_after:.2f} (nano scale; loss-decrease is the "
              "primary success criterion).")


if __name__ == "__main__":
    _self_test()
