"""Single-attempt generation with explicit budgets and lossless audit records."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from rmcq.backends.base import GenParams
from rmcq.config import SEED, TORCH_DTYPE, VLLM_DETERMINISTIC, VLLM_MAX_NUM_SEQS


def final_budget(model: str, role: str) -> int:
    if model.startswith("gpt-5"):
        return 4000
    if role == "judge":
        return 128 if model == "phi2" else 512
    if model == "phi2":
        return 768 if role == "reflection" else 512
    return 4096 if model.startswith("deepseek-r1") else 1024


def generate_once(backend, path, prompts, max_tokens, batch_size, fresh, description,
                  temperature, stop=(), adapt_to_context=False):
    from run_experiment import load_jsonl, save_jsonl

    cached = {} if fresh or not path.exists() else {r["key"]: r for r in load_jsonl(path)}
    groups = defaultdict(list)
    max_len = getattr(backend, "max_len", None)
    api_limits = {}
    if type(backend).__name__ == "AzureBackend":
        # Empty completions belong in the outcome audit for this profile.
        # Infrastructure/API failures still propagate from the backend.
        backend.audit_empty_outputs = True
        request = backend.build_kwargs("", GenParams(max_new_tokens=max_tokens, temperature=temperature))
        api_limits = {k: request[k] for k in ("max_completion_tokens", "max_tokens", "reasoning_effort") if k in request}
        max_tokens = request.get("max_completion_tokens", request.get("max_tokens", 0))
        if not max_tokens:
            raise ValueError("The final protocol requires an explicit Azure completion limit")
    for key, prompt in prompts.items():
        prompt_tokens = (
            len(backend.render_token_ids(backend.tokenizer, prompt))
            if hasattr(backend, "tokenizer") else backend.count_tokens(prompt)
        )
        capacity = max_len - prompt_tokens if max_len else max_tokens
        effective = min(max_tokens, capacity) if adapt_to_context else max_tokens
        # Limit the number of batches when Phi-2 has different available capacity.
        if adapt_to_context and 64 <= effective < max_tokens:
            effective = effective // 64 * 64
        payload = {
            "policy": "final-single-attempt-v1", "model": backend.key,
            "repo": backend.spec.repo_id, "backend": type(backend).__name__,
            "template_kwargs": backend.template_kwargs(), "prompt": prompt,
            "temperature": temperature, "stop": stop, "budget": effective,
            "context": max_len, "azure_budget": getattr(backend, "max_tokens", None),
            "api_limits": api_limits,
            "seed": SEED, "dtype": TORCH_DTYPE,
            "vllm_deterministic": VLLM_DETERMINISTIC, "vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if cached.get(key, {}).get("prompt_hash") == digest:
            continue
        meta = {
            "key": key, "prompt_hash": digest, "prompt_tokens": prompt_tokens,
            "requested_max_tokens": max_tokens, "max_new_tokens_used": max(0, effective),
            "budget_reduced_for_context": effective < max_tokens,
            "model_context_tokens": max_len, "backend": type(backend).__name__,
            "generation_profile": "final", "attempts": 0,
            "api_limits": api_limits,
        }
        if capacity < effective or effective <= 0:
            cached[key] = {
                **meta, "text": "", "raw_text": "", "visible_text": "",
                "completion_tokens": 0, "finish_reason": "prompt_context_exceeded",
                "discarded": True, "discard_reason": "prompt_plus_output_exceeds_context",
                "thinking_removed": False, "partial_think": False,
            }
        else:
            groups[effective].append((key, prompt, meta))
    save_jsonl(path, cached.values())
    print(f"{description}: {sum(map(len, groups.values()))} pending, single attempt", flush=True)
    for effective, entries in sorted(groups.items(), reverse=True):
        for start in range(0, len(entries), batch_size):
            batch = entries[start:start + batch_size]
            generations = backend.generate(
                [prompt for _, prompt, _ in batch],
                GenParams(max_new_tokens=effective, temperature=temperature, stop=stop, seed=SEED),
                desc=description,
            )
            if len(generations) != len(batch):
                raise RuntimeError("Backend returned a different number of generations than prompts")
            for (key, _, meta), gen in zip(batch, generations):
                reason = gen.finish_reason
                if reason == "length":
                    reason = "length_exhausted"
                elif not gen.text and reason != "content_filter":
                    reason = "empty_exhausted"
                discarded = reason in {"length_exhausted", "empty_exhausted", "content_filter"}
                cached[key] = {
                    **meta, "attempts": 1, "text": "" if discarded else gen.text,
                    "visible_text": gen.text, "raw_text": gen.raw_text,
                    "thinking_removed": gen.thinking_removed, "partial_think": gen.partial_think,
                    "prompt_tokens": gen.prompt_tokens or meta["prompt_tokens"],
                    "completion_tokens": gen.completion_tokens,
                    "finish_reason": reason, "raw_finish_reason": gen.finish_reason,
                    "discarded": discarded, "latency_s": gen.latency_s,
                }
            save_jsonl(path, cached.values())
    return {key: cached[key] for key in prompts}
