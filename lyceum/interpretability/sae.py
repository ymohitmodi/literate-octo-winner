"""Sparse autoencoders (SAEs) + activation steering.

This is the Frontier manual's mechanistic-interpretability toolkit at nano
scale, and it makes a surprisingly strong demo on CPU:

  1. **Collect activations** - run the model over some text and read off the
     residual-stream activation at a chosen layer. These dense vectors live in
     *superposition*: more features than neurons, packed as overlapping linear
     combinations, so individual neurons are polysemantic and hard to interpret.

  2. **Train a sparse autoencoder** - an *overcomplete* dictionary
     (``n_features`` > ``dim``) with a ReLU encoder and a linear decoder,
     trained to reconstruct the activations while an **L1 penalty** forces the
     hidden code to be sparse. Sparsity is what pushes the learned features
     toward *monosemanticity*: each feature fires for one human-legible thing.

  3. **Read the features** - inspect which dictionary features are most active
     and how sparse the code is (the L0 / fraction-active diagnostics).

  4. **Steer** - features are *directions* in activation space, so we can add a
     feature's decoder column back into the residual stream during a forward
     pass (via a hook on a block) and watch generation shift. This is
     activation steering / representation engineering, in miniature.

Everything is plain PyTorch and runs in seconds on a nano model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LyceumConfig
from ..data.tokenizer import BPETokenizer
from ..model.transformer import LyceumLM


# --------------------------------------------------------------------------- #
# 1. Collect residual-stream activations.
# --------------------------------------------------------------------------- #
def _encode_text(tok: BPETokenizer, text: str, max_len: int) -> list[int]:
    bos = tok.id("<bos>")
    eos = tok.id("<eos>")
    ids = [bos] + tok.encode(text) + [eos]
    return ids[:max_len]


@torch.no_grad()
def collect_activations(model: LyceumLM, tok: BPETokenizer, texts: list[str],
                        layer: int | None = None, device=None) -> torch.Tensor:
    """Run ``model`` over ``texts`` and return residual-stream activations.

    If ``layer is None`` we take the final normalized hidden state from
    ``model.hidden_states`` (the residual stream the LM head reads). Otherwise
    we hook block ``layer`` and capture its *output* residual stream. Returns a
    tensor of shape ``(N, dim)`` where ``N`` is the total number of (text,
    position) activations - one row per token across all texts."""
    device = device or next(model.parameters()).device
    model.eval()
    max_len = model.cfg.max_seq_len
    chunks: list[torch.Tensor] = []

    if layer is None:
        for text in texts:
            ids = torch.tensor([_encode_text(tok, text, max_len)],
                               dtype=torch.long, device=device)
            h, _ = model.hidden_states(ids)          # (1, T, dim)
            chunks.append(h[0])
    else:
        block = model.blocks[layer]
        captured: dict[str, torch.Tensor] = {}

        def hook(_module, _inp, out):
            captured["h"] = out.detach()

        handle = block.register_forward_hook(hook)
        try:
            for text in texts:
                ids = torch.tensor([_encode_text(tok, text, max_len)],
                                   dtype=torch.long, device=device)
                model.hidden_states(ids)
                chunks.append(captured["h"][0])      # (T, dim)
        finally:
            handle.remove()

    return torch.cat(chunks, dim=0)                  # (N, dim)


# --------------------------------------------------------------------------- #
# 2. Sparse autoencoder.
# --------------------------------------------------------------------------- #
class SparseAutoencoder(nn.Module):
    """Overcomplete sparse dictionary over residual-stream activations.

    ``encode(x) = ReLU(W_enc (x - b_dec) + b_enc)`` gives a sparse, non-negative
    feature code; ``decode(f) = W_dec f + b_dec`` reconstructs the activation.
    A pre-encoder bias subtraction (``b_dec``) is the standard SAE trick: it
    centers the input so the dictionary models deviations from the mean
    activation."""

    def __init__(self, dim: int, n_features: int):
        super().__init__()
        self.dim = dim
        self.n_features = n_features
        self.encoder = nn.Linear(dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, dim, bias=True)
        self.b_dec = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.encoder.weight, std=0.02)
        nn.init.zeros_(self.encoder.bias)
        # tie decoder init to encoder transpose, a common SAE warm start
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())
            self.decoder.bias.zero_()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x - self.b_dec))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.decoder(f) + self.b_dec

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        recon = self.decode(f)
        return recon, f

    def feature_direction(self, feature_idx: int) -> torch.Tensor:
        """The residual-stream direction for ``feature_idx`` (decoder column)."""
        return self.decoder.weight[:, feature_idx].detach()


def train_sae(acts: torch.Tensor, n_features: int = 256, steps: int = 300,
              l1: float = 1e-3, lr: float = 1e-3, batch_size: int = 256,
              seed: int = 0, on_log=None) -> tuple[SparseAutoencoder, list[dict]]:
    """Train an SAE to reconstruct ``acts`` (N, dim) under an L1 sparsity penalty.

    Loss = MSE reconstruction + ``l1`` * mean L1 of the feature code. The L1 term
    is what creates a sparse, more interpretable dictionary out of the dense,
    superposed residual stream. Returns ``(sae, history)``."""
    dim = acts.shape[1]
    device = acts.device
    sae = SparseAutoencoder(dim, n_features).to(device)
    # initialize the pre-encoder bias at the data mean (standard SAE practice)
    with torch.no_grad():
        sae.b_dec.copy_(acts.mean(dim=0))
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = acts.shape[0]
    history: list[dict] = []

    sae.train()
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), generator=gen).to(device)
        x = acts[idx]
        recon, f = sae(x)
        recon_loss = F.mse_loss(recon, x)
        l1_loss = f.abs().mean()
        loss = recon_loss + l1 * l1_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % max(1, steps // 6) == 0 or step == steps - 1:
            with torch.no_grad():
                frac_active = (f > 1e-6).float().mean().item()
            rec = {
                "step": step,
                "loss": round(loss.item(), 5),
                "recon": round(recon_loss.item(), 5),
                "l1": round(l1_loss.item(), 5),
                "frac_active": round(frac_active, 4),
            }
            history.append(rec)
            if on_log:
                on_log(rec)
            else:
                print(f"  [sae] step {step:>4} | loss {rec['loss']:.5f} | "
                      f"recon {rec['recon']:.5f} | l1 {rec['l1']:.5f} | "
                      f"active {rec['frac_active']*100:5.2f}%")
    return sae, history


# --------------------------------------------------------------------------- #
# 3. Read the learned features.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def top_features(sae: SparseAutoencoder, acts: torch.Tensor,
                 k: int = 10) -> dict:
    """Report the most-active features and the code's sparsity.

    Returns a dict with the top-``k`` feature indices by mean activation, the
    mean number of active features per token (L0), and the fraction of the
    dictionary that ever fires."""
    sae.eval()
    f = sae.encode(acts)                          # (N, n_features)
    active = f > 1e-6
    mean_act = f.mean(dim=0)                       # (n_features,)
    vals, idx = torch.topk(mean_act, min(k, sae.n_features))
    l0 = active.float().sum(dim=1).mean().item()   # avg active features / token
    dead = (active.sum(dim=0) == 0).float().mean().item()  # never-firing fraction
    return {
        "top_feature_ids": idx.tolist(),
        "top_feature_mean_act": [round(v, 5) for v in vals.tolist()],
        "avg_active_features_per_token_L0": round(l0, 3),
        "fraction_features_alive": round(1.0 - dead, 4),
        "n_features": sae.n_features,
    }


# --------------------------------------------------------------------------- #
# 4. Activation steering.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _greedy_generate(model: LyceumLM, tok: BPETokenizer, prompt: str,
                     max_new_tokens: int, device) -> str:
    """Minimal greedy decode using the plain forward (no KV cache), so a forward
    hook installed on a block applies to every step."""
    bos, eos = tok.id("<bos>"), tok.id("<eos>")
    u, a = tok.id("<user>"), tok.id("<assistant>")
    ids = [bos, u] + tok.encode(prompt) + [a]
    max_ctx = model.cfg.max_seq_len
    out: list[int] = []
    model.eval()
    for _ in range(max_new_tokens):
        inp = torch.tensor([ids[-max_ctx:]], dtype=torch.long, device=device)
        logits, _ = model(inp)                     # last-position logits
        nxt = int(logits[0, -1].argmax())
        if nxt == eos:
            break
        ids.append(nxt)
        out.append(nxt)
    return tok.decode(out).strip()


@torch.no_grad()
def steer(model: LyceumLM, tok: BPETokenizer, prompt: str,
          sae: SparseAutoencoder, feature_idx: int, strength: float,
          layer: int = 0, max_new_tokens: int = 16, device=None) -> dict:
    """Activation-steering demo.

    Adds ``strength * sae.feature_direction(feature_idx)`` to the residual
    stream (the output of block ``layer``) during generation via a forward hook,
    then compares unsteered vs steered greedy completions. Because the SAE
    decoder column *is* the residual-stream direction for that feature, pushing
    along it injects the feature - representation engineering in miniature.

    Returns ``{prompt, unsteered, steered, feature_idx, strength}``."""
    device = device or next(model.parameters()).device
    model.to(device)
    direction = sae.feature_direction(feature_idx).to(device)
    direction = direction / (direction.norm() + 1e-8)   # unit direction

    unsteered = _greedy_generate(model, tok, prompt, max_new_tokens, device)

    block = model.blocks[layer]

    def hook(_module, _inp, out):
        return out + strength * direction          # broadcast over (B, T, dim)

    handle = block.register_forward_hook(hook)
    try:
        steered = _greedy_generate(model, tok, prompt, max_new_tokens, device)
    finally:
        handle.remove()

    return {
        "prompt": prompt,
        "unsteered": unsteered,
        "steered": steered,
        "feature_idx": feature_idx,
        "strength": strength,
        "layer": layer,
        "changed": unsteered != steered,
    }


# --------------------------------------------------------------------------- #
# Self-test: nano model, no training of the LM needed.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile
    from ..config import get_config
    from ..data.corpus import build_pretrain_corpus

    print("=" * 64)
    print("sae.py self-test: collect -> train SAE -> features -> steer")
    print("=" * 64)

    cfg = get_config("nano")
    cfg.model.dim = 64
    cfg.model.n_layers = 2
    cfg.model.n_heads = 4
    cfg.model.n_kv_heads = 2
    cfg.model.max_seq_len = 64
    cfg.tokenizer.vocab_size = 512

    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = build_pretrain_corpus(f"{tmp}/corpus.txt", n_docs=60, seed=0)
        tok = BPETokenizer(special_tokens=cfg.tokenizer.special_tokens)
        tok.train(corpus_path.read_text(), vocab_size=cfg.tokenizer.vocab_size)
        vocab = tok.vocab_size
        print(f"tokenizer vocab_size={vocab}")

        model = LyceumLM(cfg.model, vocab)
        print(f"model params: {model.num_params():,}")

        sentences = [
            "The sun is a star that gives light.",
            "Once there was a happy cat named Mia.",
            "Bees make honey from flowers.",
            "Tom went to the park and found a ball.",
            "A triangle has three sides and three corners.",
            "The moon orbits the earth each month.",
        ]

        print("\n-- collecting residual-stream activations (layer 0 output) --")
        acts = collect_activations(model, tok, sentences, layer=0)
        print(f"activations: {tuple(acts.shape)}  (N tokens, dim)")

        print("\n-- training sparse autoencoder --")
        sae, _ = train_sae(acts, n_features=128, steps=120, l1=1e-3)

        print("\n-- top features / sparsity --")
        report = top_features(sae, acts, k=8)
        for k, v in report.items():
            print(f"  {k}: {v}")

        print("\n-- activation steering --")
        feat = report["top_feature_ids"][0]
        res = steer(model, tok, "Tell me a fact.", sae, feature_idx=feat,
                    strength=8.0, layer=0, max_new_tokens=12)
        print(f"  feature  : {res['feature_idx']} (strength {res['strength']})")
        print(f"  unsteered: {res['unsteered']!r}")
        print(f"  steered  : {res['steered']!r}")
        print(f"  changed  : {res['changed']}")

    print("\nOK: sae.py self-test ran without crashing.")
