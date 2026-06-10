"""Smoke tests for the advanced, GPU-aware feature modules.

These run on CPU at nano scale. GPU-only behavior is exercised through the
fallback path (the features detect no GPU and degrade gracefully), which is
exactly what we want to verify works on the target mini PC.
"""
import torch

from lyceum.config import get_config
from lyceum.data.corpus import build_pretrain_corpus, build_preference_dataset
from lyceum.data.curation import curate
from lyceum.data.tokenizer import BPETokenizer
from lyceum.data.dataset import PackedTextDataset, load_jsonl
from lyceum.model.transformer import LyceumLM
from lyceum import hardware


def _setup(tmp_path, steps=0):
    cfg = get_config("nano")
    raw = build_pretrain_corpus(tmp_path / "c.txt", n_docs=400).read_text()
    text, _ = curate(raw)
    tok = BPETokenizer(cfg.tokenizer.special_tokens)
    tok.train(text, cfg.tokenizer.vocab_size)
    return cfg, tok, text


def test_hardware_detect_and_fallback():
    caps = hardware.detect()
    assert caps.device_type in ("cpu", "cuda", "mps")
    flags = hardware.feature_flags()
    assert set(flags) >= {"amp", "flash", "compile", "multi_gpu", "quantize_int8"}
    assert hardware.recommend_preset() in ("nano", "tiny", "small", "xl")
    # report renders without error
    assert "capability report" in hardware.report()


def test_model_full_logits_api(tmp_path):
    cfg, tok, _ = _setup(tmp_path)
    m = LyceumLM(cfg.model, tok.vocab_size)
    ids = torch.tensor([tok.encode("the cat")[:8]])
    full = m.forward_logits(ids)
    assert full.shape[1] == ids.shape[1]            # logits for every position
    h, logits = m.hidden_states(ids)
    assert h.shape[-1] == cfg.model.dim


def test_quantization(tmp_path):
    from lyceum.inference.quantize import quantize_dynamic_int8, compare_quantization
    cfg, tok, _ = _setup(tmp_path)
    m = LyceumLM(cfg.model, tok.vocab_size)
    cmp = compare_quantization(m)
    assert cmp["int8_mb"] <= cmp["fp32_mb"]
    qm = quantize_dynamic_int8(m)
    assert qm is not None


def test_speculative_decoding(tmp_path):
    from lyceum.inference.speculative import SpeculativeDecoder, build_draft_model
    cfg, tok, _ = _setup(tmp_path)
    target = LyceumLM(cfg.model, tok.vocab_size)
    draft = build_draft_model(cfg.model, tok)
    dec = SpeculativeDecoder(target, draft, tok)
    ids = [tok.id("<bos>")] + tok.encode("the")
    out, stats = dec.generate(ids, max_new_tokens=8, k=4)
    assert isinstance(out, list) and "acceptance_rate" in stats


def test_paged_kv():
    from lyceum.inference.paged_kv import PagedKVCache
    c = PagedKVCache(n_layers=2, page_size=4, total_pages=16)
    c.allocate("a", 6)
    c.append("a", 2)
    st = c.stats()
    assert st["fragmentation"] == 0.0


def test_grpo_runs(tmp_path):
    from lyceum.train.grpo import run_grpo
    cfg, tok, _ = _setup(tmp_path)
    m = LyceumLM(cfg.model, tok.vocab_size)
    hist = run_grpo(m, tok, cfg, steps=5, group_size=4)
    assert isinstance(hist, list) and len(hist) >= 1


def test_dp_sgd_runs(tmp_path):
    from lyceum.train.dp_sgd import run_dp_sgd, estimate_epsilon
    cfg, tok, text = _setup(tmp_path)
    ds = PackedTextDataset.from_text(text, tok, cfg.data.seq_len)
    m = LyceumLM(cfg.model, tok.vocab_size)
    run_dp_sgd(m, ds, cfg, steps=3, clip=1.0, noise_multiplier=1.0)
    eps = estimate_epsilon(3, 1.0, 8, len(ds))
    assert eps > 0


def test_distributed_fallback():
    from lyceum.train.distributed import wrap_for_distributed, describe_parallelism_plan
    cfg = get_config("nano")
    m = LyceumLM(cfg.model, 256)
    # no process group initialized -> returns model unchanged (graceful)
    assert wrap_for_distributed(m, cfg) is m
    assert "parallel" in describe_parallelism_plan(cfg).lower()


def test_reward_and_constitutional(tmp_path):
    from lyceum.train.reward import RewardModel, train_reward_model
    from lyceum.train.constitutional import generate_constitutional_data, critique, CONSTITUTION
    cfg, tok, _ = _setup(tmp_path)
    build_preference_dataset(tmp_path / "p.jsonl", n=20)
    rows = load_jsonl(tmp_path / "p.jsonl")
    rm = RewardModel(LyceumLM(cfg.model, tok.vocab_size))
    train_reward_model(rm, rows, tok, cfg, steps=3)
    data = generate_constitutional_data(lambda p: "NO. that is dumb. " * 5,
                                        ["help me"], CONSTITUTION)
    assert data and "revised" in data[0]


def test_sae(tmp_path):
    from lyceum.interpretability.sae import collect_activations, train_sae
    cfg, tok, _ = _setup(tmp_path)
    m = LyceumLM(cfg.model, tok.vocab_size)
    acts = collect_activations(m, tok, ["the cat", "the sun", "a fox ran"])
    sae = train_sae(acts, n_features=64, steps=20)
    assert sae is not None


def test_hybrid_retrieval():
    from lyceum.memory.rag import VectorStore
    from lyceum.memory.hybrid import HybridRetriever
    vs = VectorStore()
    vs.add("the keyword XJ9000 appears here", source="a")
    vs.add("plants use sunlight to make food", source="b")
    hr = HybridRetriever(vs)
    hits = hr.search("XJ9000", k=2)
    assert any("XJ9000" in c.text for _, c in hits)


def test_continuous_batching():
    from lyceum.config import get_config
    from lyceum.data.corpus import build_pretrain_corpus
    from lyceum.data.curation import curate
    from lyceum.data.tokenizer import BPETokenizer
    from lyceum.model.transformer import LyceumLM
    from lyceum.inference.engine import InferenceEngine
    from lyceum.serving.batching import ContinuousBatchScheduler
    import tempfile, pathlib
    cfg = get_config("nano")
    with tempfile.TemporaryDirectory() as d:
        raw = build_pretrain_corpus(pathlib.Path(d) / "c.txt", n_docs=300).read_text()
        text, _ = curate(raw)
        tok = BPETokenizer(cfg.tokenizer.special_tokens)
        tok.train(text, cfg.tokenizer.vocab_size)
        eng = InferenceEngine(LyceumLM(cfg.model, tok.vocab_size), tok, cfg)
        sched = ContinuousBatchScheduler(eng, max_batch_size=4, batch_timeout_ms=30)
        sched.start()
        try:
            futs = [sched.submit("hi", max_new_tokens=4) for _ in range(4)]
            results = [f.result(timeout=30) for f in futs]
            assert len(results) == 4
        finally:
            sched.stop()
