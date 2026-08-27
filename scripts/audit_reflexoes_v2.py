# -*- coding: utf-8 -*-
"""Auditoria das reflexoes v2 (results/reflection_v2/reflections) vs v1 (results/reflections)."""
import json, re, csv, statistics as st, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT/"results"/"reflection_v2"/"reflections"
V1 = ROOT/"results"/"reflections"
MODELS = ["phi4-mini","llama3-8b"]; DEPTHS=["simple","complex"]; DATASETS=["arc","logiqa2"]

VAGUE=[r"be (more )?careful",r"more carefully",r"think harder",r"pay (more |closer )?attention",
r"double[- ]check",r"(be|more) thorough",r"take (my|the) time",r"read the question carefully",
r"consider all (possible )?(factors|options|possibilities|angles)",
r"be more (humble|diligent|attentive|rigorous|comprehensive|systematic|explicit|mindful|nuanced)",
r"strive to",r"in the future,? I (will|would|should)",r"for future (questions|problems|reasoning)",
r"continue to",r"critical thinking",r"improve my (reasoning|problem[- ]solving|analysis)"]
VR=[re.compile(v,re.I) for v in VAGUE]
LETTER=re.compile(r"\b(?:option|answer|choice)s?\s*\(?[A-E]\)?(?![a-z])|\b[A-E]\)\s")
GOLDCLAIM=re.compile(r"(?:correct|right) (?:answer|option|choice)\s*(?:is|was)\s*\(?([A-E])\)?",re.I)
LESSON=re.compile(r"^\s*lesson\s*:\s*(.+)",re.I|re.M)

def sents(t): return [s for s in re.split(r"(?<=[.!?])\s+",t.strip()) if s.strip()]

def audit(base, tag):
    rows=[]; samples={}
    for m in MODELS:
        for dep in DEPTHS:
            for ds in DATASETS:
                p = base/f"{m}__{m}__{dep}"/f"{ds}.jsonl"
                if not p.exists(): continue
                recs=[json.loads(l) for l in p.open(encoding="utf-8")]
                texts=[(r["uid"],(r.get("reflection_text") or "").strip(),
                        bool((r.get("extra") or {}).get("source_was_correct")), r.get("gold")) for r in recs]
                n=len(texts); wc=[len(t.split()) for _,t,_,_ in texts]
                # formato
                ends_lesson=sum(1 for _,t,_,_ in texts if t.splitlines() and t.splitlines()[-1].strip().lower().startswith("lesson:"))
                has_lesson=sum(1 for _,t,_,_ in texts if LESSON.search(t))
                # licao isolada
                lessons=[LESSON.findall(t)[-1].strip() if LESSON.search(t) else "" for _,t,_,_ in texts]
                lw=[len(l.split()) for l in lessons if l]
                lesson_vaga=sum(1 for l in lessons if l and any(r.search(l) for r in VR))
                lesson_letra=sum(1 for l in lessons if l and LETTER.search(l))
                # corpo
                vaga=sum(1 for _,t,_,_ in texts if any(r.search(t) for r in VR))
                letra=sum(1 for _,t,_,_ in texts if LETTER.search(t))
                gold=sum(1 for _,t,_,_ in texts if GOLDCLAIM.search(t))
                gold_ok=sum(1 for _,t,_,g in texts if GOLDCLAIM.search(t) and (GOLDCLAIM.search(t).group(1).upper()==(g or "").upper()))
                trunc=sum(1 for w in wc if w>120)
                dup=collections.Counter(l for l in lessons if l)
                rep=sum(c for l,c in dup.items() if c>=3)
                rows.append(dict(versao=tag,modelo=m,prof=dep,dataset=ds,n=n,
                    w_med=int(st.median(wc)), w_p90=int(sorted(wc)[int(.9*(n-1))]),
                    pct_erro=round(100*sum(1 for _,_,c,_ in texts if not c)/n,1),
                    pct_termina_lesson=round(100*ends_lesson/n,1),
                    pct_tem_lesson=round(100*has_lesson/n,1),
                    licao_w_med=int(st.median(lw)) if lw else 0,
                    pct_licao_vaga=round(100*lesson_vaga/max(has_lesson,1),1),
                    pct_licao_com_letra=round(100*lesson_letra/max(has_lesson,1),1),
                    pct_corpo_vago=round(100*vaga/n,1),
                    pct_corpo_letra=round(100*letra/n,1),
                    pct_afirma_gab=round(100*gold/n,1),
                    gab_acertado=f"{gold_ok}/{gold}",
                    pct_truncado_120=round(100*trunc/n,1),
                    pct_licao_repetida=round(100*rep/max(has_lesson,1),1)))
                samples[(tag,m,dep,ds)]=(texts,lessons,dup)
    return rows,samples

r2,s2 = audit(V2,"v2")
r1,s1 = audit(V1,"v1")
allrows = r2+r1
out=ROOT/"results"/"auditoria_reflexoes_v2.csv"
with out.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)

cols=list(allrows[0].keys())
print(" | ".join(cols))
for r in allrows: print(" | ".join(str(r[c]) for c in cols))
print("\nCSV:",out)

print("\n===== LICOES MAIS REPETIDAS (v2) =====")
for key,(texts,lessons,dup) in s2.items():
    print(f"\n--- {key[1]}/{key[2]}/{key[3]} (n={len(texts)}, licoes distintas={len(dup)}) ---")
    for l,c in dup.most_common(3):
        print(f"  [{c}x, {100*c/len(texts):.1f}%] {l[:200]}")
