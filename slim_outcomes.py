"""Recorte leve de all_outcomes.jsonl, lido direto do pacote de handoff `finish`.

Uso: python slim_outcomes.py [experiment_id]

Escreve data/results/reflection_top1/<id>/analysis/outcomes_slim.jsonl com os campos
usados pela analise (incluindo audit_flags ja consolidado por rmcq.analysis.outcome_flags),
sem os textos de resposta. Serve para abrir os notebooks sem materializar o all_outcomes.jsonl
completo; `python experiment_ops.py restore finish` continua sendo a restauracao oficial.
"""
import json, gzip, tarfile, io, time, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from rmcq.analysis import outcome_flags
EXP = sys.argv[1] if len(sys.argv) > 1 else "f8009a56a83b"
src = ROOT/"experiment_handoff"/EXP/"finish"
man = json.loads((src/"bundle.json").read_text())
class Chain(io.RawIOBase):
    def __init__(self, paths):
        self.paths=list(paths); self.i=0; self.f=open(self.paths[0],'rb'); self.done=False
    def readable(self): return True
    def readinto(self, b):
        while not self.done:
            n=self.f.readinto(b)
            if n: return n
            self.f.close(); self.i+=1
            if self.i>=len(self.paths):
                self.done=True; return 0
            self.f=open(self.paths[self.i],'rb')
        return 0
KEEP=("model","dataset","condition","val_uid","source_uid","eval_uid","eval_split","race_subset",
      "article_uid","source_article_uid","similarity","correct","eval_method","finish_reason",
      "selected_answer","embedding_truncated","reflection_finish_reason","source_answer_finish_reason")
outdir = ROOT/"data/results/reflection_top1"/EXP/"analysis"
outdir.mkdir(parents=True, exist_ok=True)
t=time.time(); n=0
stream=io.BufferedReader(Chain([src/p["name"] for p in man["parts"]]), 1<<22)
tmp = outdir/"outcomes_slim.jsonl.tmp"
with gzip.GzipFile(fileobj=stream) as z, tarfile.open(fileobj=z, mode="r|") as tar:
    for info in tar:
        if info.name.endswith("accuracy.csv"):
            (outdir/"accuracy.csv").write_bytes(tar.extractfile(info).read())
        elif info.name.endswith("content_filter_audit.jsonl"):
            (outdir/"content_filter_audit.jsonl").write_bytes(tar.extractfile(info).read())
        elif info.name.endswith("all_outcomes.jsonl"):
            fo=tar.extractfile(info)
            with tmp.open("w", encoding="utf-8") as out:
                for line in fo:
                    r=json.loads(line)
                    slim={k:r.get(k) for k in KEEP}
                    slim["audit_flags"]=outcome_flags(r)
                    out.write(json.dumps(slim, ensure_ascii=False, separators=(",",":"))+"\n")
                    n+=1
            break
tmp.replace(outdir/"outcomes_slim.jsonl")
print("rows", n, "elapsed", round(time.time()-t,1), "size", (outdir/"outcomes_slim.jsonl").stat().st_size)
