#!/usr/bin/env python
"""Roda, numa GPU, tudo o que der para rodar de um split — e nada mais.

    python run_local.py --split validation --gpu 3
    python run_local.py --split test       --gpu 5

Um comando por split, uma GPU cada, os dois ao mesmo tempo no mesmo checkout.
Cada um percorre prepare -> self-eval -> teacher -> finish -> consolidação e
para no fim; o que já estiver em disco é pulado, então repetir o comando depois
de uma interrupção retoma de onde parou em vez de recomeçar.

O professor externo (GPT-5.4) fica de fora por padrão: ele fala por API e nem
sempre há credencial ou rede. Os quatro professores abertos carregam pesos e
rodam aqui. O que ele deveria ensinar continua aparecendo como lacuna em
`gaps.json`, e `--with-external-teacher` acrescenta essa etapa quando der.

`--dry-run` imprime o plano e a conta do que falta, sem carregar modelo nenhum.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".run_state"

sys.path.insert(0, str(ROOT))
from rmcq import grid  # noqa: E402
from rmcq import results_store as store  # noqa: E402

SPLITS = ("validation", "test")


def id_path(split: str) -> Path:
    """O id fica separado por split, senão os dois comandos se atropelam.

    `.run_state/experiment_id` é global e é o que `experiment_ops` usa; dois
    runs simultâneos no mesmo checkout sobrescreveriam um ao outro ali.
    """
    return STATE / f"local_{split}_experiment_id"


def log_path(split: str) -> Path:
    return STATE / f"local_{split}.log"


def prepare_command(split: str, gpu: str) -> list[str]:
    """Argumentos que congelam a grade final neste split."""
    command = [
        sys.executable, "-u", "run_experiment.py", "prepare",
        "--gpu", gpu, "--backend", "vllm",
        "--models", ",".join(grid.STUDENTS),
        "--teachers", ",".join(grid.TEACHERS),
        "--teacher-role", "teacher-only",
        "--generation-profile", "final",
        "--eval-split", split,
        "--write-id", str(id_path(split)),
    ]
    if split == "validation":
        # O preset congela split, papel do professor e perfil de geração, e é
        # o que `validation_ops` reconhece depois.
        command += ["--preset", "validation-threshold"]
    return command


def read_id(split: str) -> str | None:
    path = id_path(split)
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    return None


def exchange_of(experiment_id: str) -> Path:
    return ROOT / "experiment_exchange" / experiment_id


def results_of(experiment_id: str) -> Path:
    return ROOT / "data/results/reflection_top1" / experiment_id


def teachers_to_run(with_external: bool) -> list[str]:
    return list(grid.TEACHERS) if with_external else list(grid.OPEN_TEACHERS)


def run(command: list[str], log, label: str) -> None:
    header = f"\n{'=' * 70}\n{label}\n{' '.join(command)}\n{'=' * 70}\n"
    print(header, flush=True)
    log.write(header)
    log.flush()
    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f"FALHOU em '{label}' (código {result.returncode}). Veja {log.name}")


def pending_summary(split: str) -> str:
    """O que a grade ainda pede, lido do arquivo consolidado."""
    path = store.gaps_path(ROOT, split)
    if not path.exists():
        return "ainda não consolidado"
    holes = json.loads(path.read_text(encoding="utf-8"))
    return (f"{holes['rows_missing_total']:,} linha(s) em "
            f"{len(holes['cells_missing'])} célula(s) vazia(s) e "
            f"{len(holes['cells_partial'])} parcial(is)")


def describe_plan(split: str, gpu: str, teachers: list[str]) -> None:
    print(f"Split ..........: {split}")
    print(f"GPU ............: {gpu}")
    print(f"Alunos .........: {len(grid.STUDENTS)} ({', '.join(grid.STUDENTS)})")
    print(f"Professores ....: {len(teachers)} ({', '.join(teachers)})")
    fora = [t for t in grid.TEACHERS if t not in teachers]
    if fora:
        print(f"Fora desta rodada: {', '.join(fora)} — continua como lacuna em gaps.json")
    experiment_id = read_id(split)
    print(f"Run ............: {experiment_id or '(ainda será criado no prepare)'}")
    if experiment_id:
        exchange = exchange_of(experiment_id)
        com_treino = sorted(p.parent.name for p in exchange.glob("students/*/train.jsonl"))
        faltando = [m for m in grid.STUDENTS if m not in com_treino]
        print(f"  com treino ...: {len(com_treino)}/{len(grid.STUDENTS)}"
              + (f" | falta gerar: {', '.join(faltando)}" if faltando else ""))
        feitos = [f"{t}/{s}" for t in teachers for s in grid.students_for(t)
                  if (exchange / "teacher" / t / "student_reflections" / f"{s}.jsonl").exists()]
        total = sum(len(grid.students_for(t)) for t in teachers)
        print(f"  pares prof. ..: {len(feitos)}/{total} já refletidos")
    print(f"Falta na grade .: {pending_summary(split)}")


ETAPAS = ("prepare", "self-eval", "teacher", "finish")


def receipts(split: str, experiment_id: str) -> list[tuple[str, Path]]:
    exchange, results = exchange_of(experiment_id), results_of(experiment_id)
    return [("prepare", exchange / "prepare_receipt.json"),
            ("self-eval", results / "self_eval/self_eval_receipt.json"),
            ("teacher", exchange / "teacher_receipt.json"),
            ("finish", results / "finish_receipt.json")]


def status() -> None:
    """Onde cada split está agora: vivo ou não, em que etapa, e o que falta.

    A pergunta que isto responde é "já acabou?". Sem ela a resposta dependia de
    cruzar log, lista de processos e recibos na mão — e um estágio que trabalha
    na CPU some do nvtop sem ter terminado, que é como dá para achar que um run
    morreu quando ele está só numa fase sem GPU.
    """
    import subprocess as sp
    agora = time.time()
    for split in SPLITS:
        print(f"\n{'=' * 62}\n{split}\n{'=' * 62}")
        vivo = sp.run(["pgrep", "-f", f"run_local.py --split {split}"],
                      capture_output=True, text=True).stdout.split()
        log = log_path(split)
        idade = (agora - log.stat().st_mtime) / 60 if log.exists() else None
        if vivo:
            print(f"  estado ....: RODANDO (PID {', '.join(vivo)})")
        else:
            print("  estado ....: parado")
        if idade is not None:
            aviso = "  <- sem escrever há muito tempo; pode estar numa fase de CPU" if idade > 45 else ""
            print(f"  último log : há {idade:.0f} min{aviso}")
        experiment_id = read_id(split)
        if not experiment_id:
            print("  run .......: ainda não criado")
            continue
        print(f"  run .......: {experiment_id}")
        for nome, caminho in receipts(split, experiment_id):
            if not caminho.exists():
                marca = "pendente"
            else:
                dados = json.loads(caminho.read_text(encoding="utf-8"))
                falta = dados.get("missing_pairs") or dados.get("teacher_arms_pending") or []
                marca = "completa" if dados.get("complete") else "parcial"
                if falta:
                    marca += f" ({len(falta)} par(es) sem o professor externo)"
                marca += f" — {time.strftime('%d/%m %H:%M', time.localtime(caminho.stat().st_mtime))}"
            print(f"    {nome:10} {marca}")
        consolidado = store.manifest_path(ROOT, split)
        if consolidado.exists():
            m = json.loads(consolidado.read_text(encoding="utf-8"))
            print(f"  consolidado: {m['rows']:,} linhas, em {m['built_at_utc']}")
        print(f"  falta .....: {pending_summary(split)}")
    print(f"\nTerminou quando `finish` estiver completa e o consolidado for reescrito.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true",
                        help="Diz onde cada split está agora e sai. Não precisa de --split nem --gpu.")
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--gpu")
    parser.add_argument("--with-external-teacher", action="store_true",
                        help=f"Inclui {grid.EXTERNAL_TEACHER}; exige credencial Azure e rede.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Imprime o plano e sai, sem carregar modelo nenhum.")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--force-preflight", dest="skip_preflight", action="store_false",
                        help="Repete o preflight mesmo num run já preparado.")
    args = parser.parse_args()
    grid.validate()
    if args.status:
        status()
        return
    if not args.split or not args.gpu:
        parser.error("--split e --gpu são obrigatórios (ou use --status)")
    teachers = teachers_to_run(args.with_external_teacher)

    print(f"\n{'#' * 70}\n# {args.split} na GPU {args.gpu}\n{'#' * 70}")
    describe_plan(args.split, args.gpu, teachers)
    if args.dry_run:
        print("\n--dry-run: nada foi executado.")
        return

    STATE.mkdir(parents=True, exist_ok=True)
    # Um lock por split: dois comandos diferentes convivem, dois iguais não.
    import fcntl
    lock = os.open(STATE / f"local_{args.split}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"Já existe um run local de {args.split} neste checkout. "
                         f"Veja {log_path(args.split)}") from None

    inicio = time.time()
    with log_path(args.split).open("a", encoding="utf-8") as log:
        log.write(f"\n\n########## {args.split} na GPU {args.gpu} "
                  f"em {time.strftime('%Y-%m-%d %H:%M:%S')} ##########\n")
        # O preflight testa os nove modelos um a um: ~15 min. Faz sentido antes
        # de começar, e é desperdício em toda retomada — se o prepare deste run
        # já fechou, estes mesmos modelos já carregaram nesta máquina.
        experiment_id = read_id(args.split)
        ja_preparado = bool(experiment_id) and (
            exchange_of(experiment_id) / "prepare_receipt.json").exists()
        if args.split == "validation" and not args.skip_preflight and not ja_preparado:
            run([sys.executable, "-u", "validation_preflight.py", "--gpu", args.gpu],
                log, "preflight: GPU, pesos e versões")
        elif args.split == "validation" and ja_preparado:
            print("preflight pulado: o prepare deste run já passou por ele "
                  "(use --force-preflight para repetir)", flush=True)

        run(prepare_command(args.split, args.gpu), log,
            "prepare: recuperação, respostas de treino e autorreflexões")
        experiment_id = read_id(args.split)
        if not experiment_id:
            raise SystemExit("prepare não gravou o id do run")
        print(f"run: {experiment_id}", flush=True)

        comum = ["--experiment-id", experiment_id, "--gpu", args.gpu]
        run([sys.executable, "-u", "run_experiment.py", "self-eval", *comum], log,
            "self-eval: baseline e autorreflexão")
        run([sys.executable, "-u", "run_experiment.py", "teacher", *comum,
             "--only-teachers", ",".join(teachers)], log,
            f"teacher: reflexões de {len(teachers)} professor(es)")
        run([sys.executable, "-u", "run_experiment.py", "finish", *comum], log,
            "finish: condições de professor")
        run([sys.executable, "consolidate_results.py", "build", "--split", args.split], log,
            "consolidação em results_definitivos")
        if args.split == "validation":
            run([sys.executable, "analyze_validation.py"], log, "tabelas e figuras")
    os.close(lock)

    minutos = (time.time() - inicio) / 60
    print(f"\n{'#' * 70}")
    print(f"# {args.split} concluído em {minutos:.0f} min — run {experiment_id}")
    print(f"# Falta na grade: {pending_summary(args.split)}")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
