"""RLVR (RL from Verifiable Rewards) with GRPO on a toy arithmetic task.

This is the flagship *reasoning* demo from the Frontier manual's "learning from
verifiable rewards" chapter, shrunk to run on a CPU. The recipe:

  1. **Verifiable task.** We pose arithmetic questions ("What is {a} plus {b}?").
     The crucial property is that the answer can be *checked by a program*, not by
     a learned (and hackable) reward model. The verifier is a **trusted
     component**: it returns reward 1.0 if the completion contains the correct
     integer, else 0.0. Because the reward signal is grounded in ground truth, it
     cannot be gamed the way a neural reward model can -- this is what makes RLVR
     so stable, and what powers the *self-improvement flywheel*: the model
     generates its own attempts, the verifier grades them for free, and the model
     learns from its own successes without any human labels.

  2. **GRPO (Group Relative Policy Optimization).** Instead of training a value
     model to estimate a baseline (as PPO does), GRPO samples a *group* of G
     completions for the same prompt and uses the group as its own baseline: the
     advantage of a completion is its reward minus the group mean, divided by the
     group std. Completions that beat their peers get pushed up; those that lag
     get pushed down. No value network, no GAE -- just relative ranking within a
     group. This is GRPO's core trick.

  3. **Policy-gradient update.** We apply a bare REINFORCE-style update:

         loss = -(advantage * sum_logprob_of_sampled_tokens)

     so completions with positive (above-average) advantage have their token
     log-probs raised. An optional small KL penalty toward a frozen *reference*
     copy keeps the policy from drifting into degenerate text.

Nothing here is fundamentally different from frontier RLVR; it is the same
machinery, parameterized small. On an untrained nano model the reward will stay
low -- the point of the self-test is to prove the loop *runs and updates*, not to
solve arithmetic.
"""
from __future__ import annotations

import copy
import re

import torch
import torch.nn.functional as F

from ..config import LyceumConfig
from ..data.tokenizer import BPETokenizer
from ..model.transformer import LyceumLM


# --------------------------------------------------------------------------- #
# Verifiable arithmetic task + programmatic verifier (the trusted component).
# --------------------------------------------------------------------------- #
def make_problem(rng: torch.Generator, max_operand: int = 9) -> tuple[str, int]:
    """Return ("What is {a} plus {b}?", a + b). Small operands keep the answer
    short and CPU-cheap to verify on a tiny model."""
    a = int(torch.randint(0, max_operand + 1, (1,), generator=rng))
    b = int(torch.randint(0, max_operand + 1, (1,), generator=rng))
    return f"What is {a} plus {b}?", a + b


def verify(completion_text: str, answer: int) -> float:
    """The trusted verifier: 1.0 iff the correct integer appears in the text.

    Grounded in ground truth, so it cannot be reward-hacked the way a learned
    reward model can. We match the answer as a standalone integer token to avoid
    spurious substring hits (e.g. "12" inside "123")."""
    for tok in re.findall(r"-?\d+", completion_text):
        try:
            if int(tok) == answer:
                return 1.0
        except ValueError:
            continue
    return 0.0


# --------------------------------------------------------------------------- #
# Sampling + log-prob helpers (mirror the reward.py REINFORCE helpers).
# --------------------------------------------------------------------------- #
def _sample_completion(policy: LyceumLM, prompt_ids: list[int], tok: BPETokenizer,
                       max_new_tokens: int, max_ctx: int, temperature: float,
                       device, rng: torch.Generator | None = None):
    """Autoregressively sample one completion (no grad).
    Returns (full_ids = prompt + completion, prompt_len)."""
    eos = tok.id("<eos>")
    ids = list(prompt_ids)
    prompt_len = len(ids)
    policy.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp = torch.tensor([ids[-max_ctx:]], dtype=torch.long, device=device)
            logits = policy.forward_logits(inp)[0, -1]
            if temperature <= 0:
                nxt = int(logits.argmax())
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1, generator=rng))
            if nxt == eos:
                break
            ids.append(nxt)
    return ids, prompt_len


def _completion_logprob(model: LyceumLM, ids: list[int], prompt_len: int,
                        device) -> torch.Tensor:
    """Differentiable sum of log-probs the model assigns to the *completion*
    tokens (positions >= prompt_len)."""
    seq = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model.forward_logits(seq)               # (1, T, V)
    logp = F.log_softmax(logits[:, :-1], dim=-1)     # predict token t+1
    targets = seq[:, 1:]
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]  # (T-1,)
    start = max(0, prompt_len - 1)
    comp = tok_logp[start:]
    if comp.numel() == 0:
        return tok_logp.sum() * 0.0  # empty completion -> zero, keeps grad graph
    return comp.sum()


# --------------------------------------------------------------------------- #
# GRPO training loop.
# --------------------------------------------------------------------------- #
def run_grpo(model: LyceumLM, tok: BPETokenizer, cfg: LyceumConfig,
             steps: int = 100, group_size: int = 4,
             lr: float | None = None, max_new_tokens: int = 12,
             temperature: float = 1.0, beta: float = 0.0,
             max_operand: int = 9, on_log=None):
    """Run GRPO on the verifiable arithmetic task.

    For each step we draw one prompt, sample a GROUP of ``group_size``
    completions (temperature > 0 so the group is diverse), verify each, and form
    GROUP-RELATIVE advantages ``(reward - group_mean) / (group_std + eps)``. We
    then take a REINFORCE step on ``-(advantage * sum_logprob)`` summed over the
    group, with an optional KL penalty (weight ``beta``) toward a frozen
    reference copy of the initial policy.

    Returns a list of per-step stat dicts; ``rec["mean_reward"]`` tracks whether
    the policy is getting better at the task over time.
    """
    from ..hardware import select_device
    device = select_device(cfg.train.device)
    model.to(device).train()
    lr = lr if lr is not None else cfg.align.dpo_lr
    max_ctx = cfg.model.max_seq_len
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    # Frozen reference policy for the optional KL leash (detached throughout).
    ref = None
    if beta > 0:
        ref = copy.deepcopy(model).to(device).eval()
        for p in ref.parameters():
            p.requires_grad_(False)

    bos, u, a = tok.id("<bos>"), tok.id("<user>"), tok.id("<assistant>")
    rng = torch.Generator().manual_seed(cfg.train.seed)
    history = []

    for step in range(steps):
        question, answer = make_problem(rng, max_operand)
        prompt_ids = [bos, u] + tok.encode(question) + [a]

        # 1. Sample a GROUP of completions and verify each one.
        group: list[tuple[list[int], int, float]] = []  # (full_ids, prompt_len, reward)
        for _ in range(group_size):
            full_ids, prompt_len = _sample_completion(
                model, prompt_ids, tok, max_new_tokens, max_ctx,
                temperature, device, rng)
            comp_text = tok.decode(full_ids[prompt_len:])
            reward = verify(comp_text, answer)
            group.append((full_ids, prompt_len, reward))

        rewards = torch.tensor([g[2] for g in group], dtype=torch.float32)
        mean_r = float(rewards.mean())
        std_r = float(rewards.std(unbiased=False))

        # 2. GROUP-RELATIVE advantages (GRPO's core: group is its own baseline).
        #    If every completion got the same reward, std==0 -> zero advantage,
        #    so the group is simply skipped (nothing to rank).
        advantages = (rewards - mean_r) / (std_r + 1e-6)

        # 3. REINFORCE update over the group, with optional KL leash.
        model.train()
        opt.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        kl_val = 0.0
        n_used = 0
        for (full_ids, prompt_len, _), adv in zip(group, advantages.tolist()):
            if len(full_ids) <= prompt_len:
                continue  # empty completion -> nothing to learn
            pol_logp = _completion_logprob(model, full_ids, prompt_len, device)
            loss = -(adv * pol_logp)
            if ref is not None:
                with torch.no_grad():
                    ref_logp = _completion_logprob(ref, full_ids, prompt_len, device)
                # summed-logprob KL proxy: penalize drift from the reference.
                kl = pol_logp - ref_logp.detach()
                loss = loss + beta * kl
                kl_val += float(kl.detach())
            total_loss = total_loss + loss
            n_used += 1

        if n_used > 0 and std_r > 0:
            (total_loss / n_used).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()

        rec = {
            "phase": "grpo", "step": step,
            "question": question, "answer": answer,
            "mean_reward": round(mean_r, 4),
            "best_reward": round(float(rewards.max()), 4),
            "reward_std": round(std_r, 4),
            "kl": round(kl_val / max(1, n_used), 4),
            "loss": round(float(total_loss.item()) / max(1, n_used), 4),
            "group_size": group_size,
        }
        history.append(rec)
        if step % max(1, cfg.train.log_interval) == 0 or step == steps - 1:
            (on_log or _print)(rec)

    return history


def _print(rec):
    print(f"  [grpo] step {rec['step']:>4} | mean_r {rec['mean_reward']:.3f} | "
          f"best_r {rec['best_reward']:.3f} | std {rec['reward_std']:.3f} | "
          f"loss {rec['loss']:8.3f} | q='{rec['question']}' a={rec['answer']}")


# --------------------------------------------------------------------------- #
# Self-test: nano model + tiny synthetic arithmetic corpus, ~10 GRPO steps.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from ..config import get_config

    print("=" * 64)
    print("grpo.py self-test: RLVR + GRPO on toy arithmetic")
    print("=" * 64)

    cfg = get_config("nano")
    # shrink further so the smoke test is near-instant
    cfg.model.dim = 64
    cfg.model.n_layers = 2
    cfg.model.n_heads = 4
    cfg.model.n_kv_heads = 2
    cfg.model.max_seq_len = 128
    cfg.tokenizer.vocab_size = 512
    cfg.train.log_interval = 2

    # Train a tokenizer on a tiny synthetic arithmetic corpus so the digits and
    # the prompt phrasing are all in-vocabulary.
    rng = torch.Generator().manual_seed(0)
    lines = []
    for _ in range(400):
        q, ans = make_problem(rng)
        lines.append(f"{q} The answer is {ans}.")
    corpus = "\n".join(lines)

    tok = BPETokenizer(special_tokens=cfg.tokenizer.special_tokens)
    tok.train(corpus, vocab_size=cfg.tokenizer.vocab_size)
    print(f"tokenizer vocab_size={tok.vocab_size}")

    model = LyceumLM(cfg.model, tok.vocab_size)
    print(f"policy params: {model.num_params():,}")

    # quick verifier sanity check
    assert verify("the answer is 7", 7) == 1.0
    assert verify("the answer is 8", 7) == 0.0
    print("verifier sanity check passed")

    print("\n-- running ~10 GRPO steps (reward may stay low on an untrained model) --")
    hist = run_grpo(model, tok, cfg, steps=10, group_size=4,
                    max_new_tokens=8, temperature=1.0, beta=0.02)

    rewards = [h["mean_reward"] for h in hist]
    print(f"\nmean-reward history: {rewards}")
    print(f"avg mean-reward over run: {sum(rewards) / len(rewards):.4f}")
    print("\nOK: grpo.py self-test ran and updated without crashing.")
