# -*- coding: utf-8 -*-
"""Auditoria da qualidade das reflexoes efetivamente consumidas em results/diagnostics."""
import json, re, os, sys, collections, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFL = ROOT / "results" / "reflections"
PAIRS = [("phi4-mini","phi4-mini"), ("llama3-8b","llama3-8b")]
DEPTHS = ["simple","complex"]
DATASETS = ["arc","logiqa2"]

VAGUE = [
 r"be more careful", r"be careful", r"more carefully", r"think harder", r"pay (more )?attention",
 r"double[- ]check", r"be more thorough", r"more thorough", r"take (my|the) time",
 r"read the question carefully", r"consider all (possible )?(factors|options|possibilities)",
 r"be more (humble|diligent|attentive|rigorous|comprehensive|systematic)",
 r"strive to", r"in the future,? I (will|would)", r"continue to",
]
VAGUE_RE = [re.compile(p, re.I) for p in VAGUE]

LETTER_RE = re.compile(r"\b(option|answer|choice)s?\s*\(?\s*[A-E]\)?\b|\b[A-E]\)\s", re.I)
GOLD_CLAIM_RE = re.compile(r"(correct answer (is|was)|right answer (is|was)|the answer (is|was))", re.I)
LESSON_RE = re.compile(r"^\s*lesson\s*:", re.I | re.M)

def load(pair, depth, ds):
    p = REFL / f"{pair[0]}__{pair[1]}__{depth}" / f"{ds}.jsonl"
    out = []
    if not p.exists(): return out
    for line in p.open(encoding="utf-8"):
        out.append(json.loads(line))
    return out

def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]

rows = []
detail = {}
for pair in PAIRS:
    for depth in DEPTHS:
        for ds in DATASETS:
            recs = load(pair, depth, ds)
            if not recs: continue
            texts = [(r["uid"], (r.get("reflection_text") or "").strip(), bool((r.get("extra") or {}).get("source_was_correct"))) for r in recs]
            wc = [len(t.split()) for _,t,_ in texts]
            n = len(texts)
            n_wrong = sum(1 for _,_,c in texts if not c)
            vague_hits = [sum(1 for rx in VAGUE_RE if rx.search(t)) for _,t,_ in texts]
            has_vague = sum(1 for v in vague_hits if v>0)
            letters = sum(1 for _,t,_ in texts if LETTER_RE.search(t))
            gold = sum(1 for _,t,_ in texts if GOLD_CLAIM_RE.search(t))
            lesson = sum(1 for _,t,_ in texts if LESSON_RE.search(t))
            # duplicidade: sentenca final e sentencas repetidas no corpus
            lastsent = collections.Counter(sentences(t)[-1] if sentences(t) else "" for _,t,_ in texts)
            allsent = collections.Counter(s for _,t,_ in texts for s in sentences(t))
            rep_sent = sum(c for s,c in allsent.items() if c>=5 and len(s.split())>=6)
            tot_sent = sum(allsent.values())
            # vocabulario: type-token nas ultimas 120 palavras (o que sobrevive ao corte)
            tails = [" ".join(t.split()[-120:]) for _,t,_ in texts]
            tail_tokens = [w.lower() for tl in tails for w in re.findall(r"[a-z']+", tl.lower())]
            ttr = len(set(tail_tokens))/max(len(tail_tokens),1)
            rows.append(dict(model=pair[0], depth=depth, dataset=ds, n=n,
                w_med=int(st.median(wc)), w_p90=int(sorted(wc)[int(.9*(n-1))]),
                pct_from_wrong=round(100*n_wrong/n,1),
                pct_vago=round(100*has_vague/n,1),
                pct_letra=round(100*letters/n,1),
                pct_afirma_gabarito=round(100*gold/n,1),
                pct_linha_lesson=round(100*lesson/n,1),
                pct_sent_repetida=round(100*rep_sent/max(tot_sent,1),1),
                top_last_sent_pct=round(100*lastsent.most_common(1)[0][1]/n,1),
                ttr_cauda=round(ttr,3)))
            detail[(pair[0],depth,ds)] = (texts, lastsent, allsent)

import csv
out = ROOT/"results"/"auditoria_reflexoes.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

hdr = list(rows[0].keys())
print(" | ".join(hdr))
for r in rows: print(" | ".join(str(r[h]) for h in hdr))
print("\nCSV:", out)

# frases de encerramento mais comuns (o que sobrevive ao corte pela cauda)
print("\n===== SENTENCAS FINAIS MAIS COMUNS =====")
for key,(texts,lastsent,allsent) in detail.items():
    print(f"\n--- {key} (n={len(texts)}) ---")
    for s,c in lastsent.most_common(3):
        print(f"  [{c}x, {100*c/len(texts):.0f}%] {s[:220]}")
