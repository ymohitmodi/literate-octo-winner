"""Knowledge distillation: teacher -> student.

Frontier-manual chapter on *how capable small models are made*. The headline
frontier models are large; the ones you actually run on a phone or a mini PC are
small. A leading way to make a small model punch above its weight is **knowledge
distillation** (Hinton et al.): instead of training the small "student" only on
the hard one-hot next token, also train it to match the large "teacher" model's
full *soft* probability distribution.

Those soft probabilities carry "dark knowledge": the teacher doesn't just say
"the next token is X", it says "X is most likely, but Y and Z are plausible and
Q is absurd". That relative structure over the whole vocabulary is a far richer
training signal than a single correct label, so the student learns faster and
generalizes better than training from scratch.

Two knobs:

  * ``temperature`` (T > 1) -- divides logits before softmax, *softening* the
    distributions so the small probabilities (the dark knowledge) become visible
    in the gradient. The KL term is scaled by ``T**2`` to keep gradient
    magnitudes comparable as T changes (standard Hinton correction).

  * ``alpha`` in [0, 1] -- blends the two objectives::

        loss = alpha * KL(student_T || teacher_T) * T**2
             + (1 - alpha) * CE(student, hard_next_token)

    ``alpha=1`` is pure distillation (match the teacher), ``alpha=0`` is ordinary
    pretraining. The default ``0.5`` keeps the student honest about real next
    tokens while still absorbing the teacher's dark knowledge.

The teacher runs under ``torch.no_grad`` via ``forward_logits``; only the student
is optimized. CPU-feasible at nano scale. Self-test below trains a small teacher
with ``pretrain``, then distills an even smaller student and shows the distill
loss decreasing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import LyceumConfig
from ..model.transformer import LyceumLM


@dataclass
class DistillState:
    history: list[dict] = field(default_factory=list)

    def losses(self) -> list[float]:
        return [h["loss"] for h in self.history]


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                      targets: torch.Tensor, temperature: float = 2.0,
                      alpha: float = 0.5, ignore_index: int = -100
                      ) -> tuple[torch.Tensor, float, float]:
    """Blend temperature-softened KL with hard-label cross-entropy.

    Shapes: logits ``(B, T, V)``, targets ``(B, T)``. Returns
    ``(loss, kl_value, ce_value)``.
    """
    B, T, V = student_logits.shape
    s = student_logits.reshape(-1, V)
    t = teacher_logits.reshape(-1, V)
    tgt = targets.reshape(-1)

    # --- soft distillation term: KL on temperature-softened distributions ---
    s_log_prob = F.log_softmax(s / temperature, dim=-1)
    t_prob = F.softmax(t / temperature, dim=-1)
    # batchmean + T**2 keeps the gradient scale comparable across temperatures
    kl = F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (temperature ** 2)

    # --- hard term: standard next-token cross-entropy (skips masked tokens) ---
    ce = F.cross_entropy(s, tgt, ignore_index=ignore_index)

    loss = alpha * kl + (1.0 - alpha) * ce
    return loss, float(kl.detach()), float(ce.detach())


def distill(teacher: LyceumLM, student: LyceumLM, dataset, tok, cfg: LyceumConfig,
            *, steps: int = 200, temperature: float = 2.0, alpha: float = 0.5,
            lr: float | None = None, batch_size: int | None = None,
            log_interval: int = 10, on_log=None) -> DistillState:
    """Train ``student`` to imitate ``teacher`` on ``dataset``.

    The teacher is frozen (eval + no grad); the student is optimized with AdamW.
    ``temperature`` and ``alpha`` are the distillation knobs documented above.
    Returns a :class:`DistillState` whose ``history`` is the per-log-step loss
    record (use ``.losses()`` to see the curve decreasing).
    """
    from ..hardware import select_device, autocast, detect
    t = cfg.train
    torch.manual_seed(t.seed)
    device = select_device(t.device)
    teacher.to(device).eval()
    student.to(device).train()
    caps = detect()
    use_amp = t.amp and caps.amp

    bs = batch_size or t.batch_size
    lr = lr if lr is not None else t.lr
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=t.weight_decay)

    state = DistillState()

    def data_iter():
        while True:
            for b in loader:
                yield b

    it = data_iter()
    for step in range(steps):
        x, y = next(it)
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            teacher_logits = teacher.forward_logits(x)

        opt.zero_grad(set_to_none=True)
        with autocast(use_amp):
            student_logits = student.forward_logits(x)
            loss, kl, ce = distillation_loss(
                student_logits, teacher_logits, y,
                temperature=temperature, alpha=alpha)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), t.grad_clip)
        opt.step()

        if step % log_interval == 0 or step == steps - 1:
            rec = {"step": step, "loss": round(float(loss.detach()), 4),
                   "kl": round(kl, 4), "ce": round(ce, 4),
                   "temperature": temperature, "alpha": alpha}
            state.history.append(rec)
            if on_log:
                on_log(rec)
            else:
                print(f"  step {step:>4} | loss {rec['loss']:7.4f} | "
                      f"kl {rec['kl']:7.4f} | ce {rec['ce']:7.4f}")
    return state


# --------------------------------------------------------------------------- #
# Fast self-test: train a small teacher, distill a smaller student.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from ..config import get_config
    from ..data.corpus import build_pretrain_corpus
    from ..data.tokenizer import BPETokenizer
    from ..data.dataset import PackedTextDataset
    from ..model.transformer import LyceumLM
    from .pretrain import pretrain

    torch.manual_seed(0)
    cfg = get_config("nano")
    # keep everything fast for a <60s CPU smoke test
    cfg.train.max_steps = 20
    cfg.train.warmup_steps = 4
    cfg.train.batch_size = 8
    cfg.train.log_interval = 10
    cfg.train.amp = False
    cfg.data.seq_len = 48
    cfg.model.max_seq_len = 48
    # shrink the teacher too so its brief pretrain stays well under budget
    cfg.model.dim = 96
    cfg.model.n_layers = 3

    with tempfile.TemporaryDirectory() as d:
        corpus_path = Path(d) / "corpus.txt"
        build_pretrain_corpus(corpus_path, n_docs=120, seed=0)
        text = corpus_path.read_text(encoding="utf-8")

        tok = BPETokenizer(special_tokens=cfg.tokenizer.special_tokens)
        tok.train(text, vocab_size=512)
        vocab = tok.vocab_size
        dataset = PackedTextDataset.from_text(text, tok, cfg.data.seq_len)
        print(f"vocab={vocab} dataset_windows={len(dataset)}")

        # --- teacher: the 'nano' model, briefly pretrained -------------------
        teacher = LyceumLM(cfg.model, vocab_size=vocab)
        print(f"\n[teacher] params={teacher.num_params():,}  pretraining...")
        pretrain(teacher, dataset, cfg)

        # --- student: strictly smaller than the teacher ----------------------
        student_mcfg = type(cfg.model)(
            dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
            max_seq_len=cfg.model.max_seq_len)
        student = LyceumLM(student_mcfg, vocab_size=vocab)
        print(f"[student] params={student.num_params():,} "
              f"(<{teacher.num_params():,} teacher)\n")

        print("[distill] temperature=2.0 alpha=0.5 (KL soft + CE hard)")
        state = distill(teacher, student, dataset, tok, cfg,
                        steps=20, temperature=2.0, alpha=0.5,
                        batch_size=8, log_interval=5)

        losses = state.losses()
        print(f"\ndistill loss: first={losses[0]:.4f} last={losses[-1]:.4f}")
        assert losses[-1] < losses[0], "distillation loss should decrease"
        print("ok: student distillation loss decreased")
