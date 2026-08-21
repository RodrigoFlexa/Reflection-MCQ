"""
Interface de linha de comando.

    python -m rmcq <comando> [opções]

Todos os comandos de geração aceitam:
  --dry-run   estima gerações e tokens sem carregar modelo
  --backend   sobrepõe RMCQ_BACKEND do .env (vllm | hf | stub)
  --limit N   processa só os N primeiros itens de cada arquivo (piloto)

Toda etapa é retomável: rodar de novo pula o que já está no JSONL.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

import rmcq  # noqa: F401  — carrega o .env antes de qualquer torch
from rmcq.config import (
    ACTIVE_MODELS,
    ALL_DATASETS,
    ALL_MODELS,
    DEPTHS,
    INACTIVE_MODELS,
    K_CANDIDATES,
    K_VALUES,
    STUDENTS,
    TEACHERS,
    ensure_dirs,
)
from rmcq.data import EXCHANGE_DIRECTIONS
from rmcq.store import get_logger, log_to_file

log = get_logger(__name__)

# Estimativas de tokens por geração, usadas só pelo --dry-run. Vêm de ordens de
# grandeza observadas em MCQ com cadeia de raciocínio; servem para dimensionar a
# execução, não para prever o custo exato.
EST_TOKENS = {
    "baseline": {"in": 180, "out": 260},
    "reflect": {"simple": {"in": 520, "out": 150}, "complex": {"in": 520, "out": 420}},
    "eval": {"in_base": 200, "in_per_reflection": {"simple": 130, "out": 260},
             "in_per_reflection_complex": 340},
    "retry": {"in": 210, "out": 260},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_plan(rows: Sequence[dict[str, Any]], est_out: float, est_in: float) -> None:
    if not rows:
        print("Nada a fazer.")
        return

    total = sum(r.get("total", 0) for r in rows)
    done = sum(r.get("done", 0) for r in rows)
    pending = sum(r.get("pending", 0) for r in rows)
    gens = sum(r.get("generations", r.get("pending", 0)) for r in rows)

    print(f"\n{'configurações':<20} {len(rows):>12,}")
    print(f"{'itens totais':<20} {total:>12,}")
    print(f"{'já concluídos':<20} {done:>12,}  ({100 * done / max(total, 1):.1f}%)")
    print(f"{'pendentes':<20} {pending:>12,}")
    if gens != pending:
        print(f"{'gerações':<20} {gens:>12,}  (amostras múltiplas por item)")
    print(f"{'tokens de entrada':<20} {est_in:>12,.0f}  (estimado)")
    print(f"{'tokens de saída':<20} {est_out:>12,.0f}  (estimado)")

    # 1500 tok/s é uma vazão conservadora de vLLM para um 8B em bfloat16 numa GPU
    # de datacenter; o backend hf fica uma ordem de grandeza abaixo.
    for name, rate in (("vLLM ~1500 tok/s", 1500), ("transformers ~150 tok/s", 150)):
        hours = est_out / rate / 3600
        print(f"{'tempo em ' + name:<20} {hours:>12,.1f} h")

    incomplete = [r for r in rows if r.get("pending")]
    if incomplete:
        print(f"\nPrimeiras configurações pendentes ({min(8, len(incomplete))} de {len(incomplete)}):")
        for r in incomplete[:8]:
            bits = [f"{k}={v}" for k, v in r.items()
                    if k in ("student", "teacher", "model", "depth", "k", "dataset", "split")]
            print(f"  {' '.join(bits):<70} {r['pending']:>7,} pendentes")


def _print_totals(per_stage: list[tuple[str, int, float, float]]) -> None:
    """Tabela agregada de custo do pipeline, uma linha por etapa."""
    print(f"\n{'etapa':<12}{'pendentes':>12}{'tok. entrada':>15}{'tok. saída':>14}"
          f"{'vLLM (h)':>11}{'hf (h)':>10}")
    print("-" * 74)
    t_pend = t_in = t_out = 0.0
    for name, pending, est_out, est_in in per_stage:
        t_pend += pending
        t_in += est_in
        t_out += est_out
        if pending == 0 and est_out == 0:
            print(f"{name:<12}{'—':>12}{'—':>15}{'—':>14}{'—':>11}{'—':>10}")
            continue
        print(f"{name:<12}{pending:>12,}{est_in:>15,.0f}{est_out:>14,.0f}"
              f"{est_out / 1500 / 3600:>11.1f}{est_out / 150 / 3600:>10.1f}")
    print("-" * 74)
    print(f"{'TOTAL':<12}{t_pend:>12,.0f}{t_in:>15,.0f}{t_out:>14,.0f}"
          f"{t_out / 1500 / 3600:>11.1f}{t_out / 150 / 3600:>10.1f}")
    print("\nEstimativas de tokens são ordens de grandeza, para dimensionar a execução.")
    print("Vazão: ~1500 tok/s é vLLM num 8B em bfloat16; ~150 tok/s é o backend hf.")


# --- estimadores por etapa, compartilhados entre os subcomandos e o run-all ---


def _est_baseline(args) -> tuple[list[dict[str, Any]], float, float]:
    from rmcq.stages import baseline

    rows = baseline.plan(args.students, args.datasets, getattr(args, "splits", None))
    p = sum(r.get("pending", 0) for r in rows)
    return rows, p * EST_TOKENS["baseline"]["out"], p * EST_TOKENS["baseline"]["in"]


def _est_reflect(args) -> tuple[list[dict[str, Any]], float, float]:
    from rmcq.stages import reflect

    rows = reflect.plan(args.students, args.teachers, args.depths, args.datasets)
    est = EST_TOKENS["reflect"]
    out = sum(r.get("pending", 0) * est[r["depth"]]["out"] for r in rows)
    inn = sum(r.get("pending", 0) * est[r["depth"]]["in"] for r in rows)
    return rows, out, inn


def _est_eval(args) -> tuple[list[dict[str, Any]], float, float]:
    from rmcq.stages import evaluate

    rows = evaluate.plan(args.students, args.teachers, args.depths, args.k, args.datasets)
    est = EST_TOKENS["eval"]
    out = sum(r.get("pending", 0) * est["in_per_reflection"]["out"] for r in rows)
    inn = sum(
        r.get("pending", 0) * (
            est["in_base"] + r["k"] * (
                est["in_per_reflection"]["simple"] if r["depth"] == "simple"
                else est["in_per_reflection_complex"]
            )
        )
        for r in rows
    )
    return rows, out, inn


def _est_retry(args) -> tuple[list[dict[str, Any]], float, float]:
    from rmcq.stages import conditions

    rows = conditions.plan_retry(args.students, args.datasets)
    p = sum(r.get("pending", 0) for r in rows)
    return rows, p * EST_TOKENS["retry"]["out"], p * EST_TOKENS["retry"]["in"]


# `index` e `analyze` não geram nada com modelo: custo de GPU zero.
ESTIMATORS = {
    "baseline": _est_baseline,
    "reflect": _est_reflect,
    "eval": _est_eval,
    "retry": _est_retry,
}

# `selfcons` foi retirado da grade: ver rmcq/stages/conditions.py. O código
# continua no repositório, mas não há caminho de execução para ele.
RUN_ALL_STAGES = ("baseline", "index", "reflect", "eval", "retry", "analyze")


def _pos_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("precisa ser positivo")
    return n


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dry-run", action="store_true", help="estimar sem executar")
    p.add_argument("--backend", choices=("vllm", "hf", "stub", "azure"), default=None,
                   help=("sobrepõe RMCQ_BACKEND do .env. Professores de API "
                         "(gpt5, gpt4) vão para o Azure de qualquer forma, "
                         "exceto sob --backend stub"))
    p.add_argument("--limit", type=_pos_int, default=None,
                   help="só os N primeiros itens de cada arquivo (piloto)")
    p.add_argument("--log-file", action="store_true", help="espelhar o log em results/logs/")


def _add_grid(p: argparse.ArgumentParser, *, students=True, teachers=False,
              depths=False, ks=False, datasets=True, splits=False) -> None:
    if students:
        p.add_argument("--students", "--models", nargs="+", default=None, metavar="KEY",
                       help=f"padrão: os ativos ({', '.join(STUDENTS)})")
    if teachers:
        p.add_argument("--teachers", nargs="+", default=None, metavar="KEY",
                       help=f"padrão: os ativos ({', '.join(TEACHERS)})")
    if depths:
        p.add_argument("--depths", nargs="+", default=None, choices=list(DEPTHS) + ["all"],
                       metavar="DEPTH", help=f"padrão: {', '.join(DEPTHS)}")
    if ks:
        p.add_argument("-k", "--k", nargs="+", type=_pos_int, default=None, metavar="K",
                       help=(f"padrão: {', '.join(map(str, K_VALUES))} "
                             f"(candidatos do Caderno: {', '.join(map(str, K_CANDIDATES))})"))
    if datasets:
        p.add_argument("--datasets", nargs="+", default=None, metavar="KEY",
                       help=f"padrão: todos ({', '.join(ALL_DATASETS)})")
    if splits:
        p.add_argument("--splits", nargs="+", default=None, metavar="SPLIT",
                       help=("padrão: train e test, os únicos que o laço consome. "
                             "'all' inclui validation, reservada para escolher "
                             "hiperparâmetros sem tocar no teste"))


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------


def cmd_setup_data(args) -> int:
    from rmcq.stages import setup_data

    return setup_data.run(args.datasets, force=args.force, verify_only=args.verify)


def cmd_setup_models(args) -> int:
    from rmcq.stages import setup_models

    return setup_models.run(args.models, tokenizers_only=args.tokenizers_only,
                            check_only=args.check)


def cmd_smoke(args) -> int:
    from rmcq.stages import setup_models

    return setup_models.smoke(args.models, args.backend)


def cmd_baseline(args) -> int:
    from rmcq.stages import baseline

    if args.dry_run:
        _print_plan(*_est_baseline(args))
        return 0
    baseline.run(args.students, args.datasets, args.splits, args.backend, args.limit)
    return 0


def cmd_reflect(args) -> int:
    from rmcq.stages import reflect

    if args.dry_run:
        _print_plan(*_est_reflect(args))
        return 0
    reflect.run(args.students, args.teachers, args.depths, args.datasets,
                args.backend, args.limit)
    return 0


def cmd_index(args) -> int:
    from rmcq.config import EMBEDDER
    from rmcq.retrieval import build

    result = build(args.datasets, embedder=args.embedder or EMBEDDER,
                   max_k=args.max_k, force=args.force)
    print(f"construídos: {result['built'] or 'nenhum'}")
    print(f"reaproveitados: {result['skipped'] or 'nenhum'}")
    return 0


def cmd_eval(args) -> int:
    from rmcq.config import EMBEDDER
    from rmcq.stages import evaluate

    if args.dry_run:
        _print_plan(*_est_eval(args))
        return 0
    evaluate.run(args.students, args.teachers, args.depths, args.k, args.datasets,
                 args.backend, args.embedder or EMBEDDER, args.limit)
    return 0


def cmd_retry(args) -> int:
    from rmcq.stages import conditions

    if args.dry_run:
        _print_plan(*_est_retry(args))
        return 0
    conditions.run_retry(args.students, args.datasets, args.backend, args.limit)
    return 0


def cmd_export_bundle(args) -> int:
    from rmcq.stages import exchange

    result = exchange.export(
        args.direction,
        students=args.students, teachers=getattr(args, "teachers", None),
        depths=getattr(args, "depths", None), datasets=args.datasets,
    )
    print(
        f"\npacote {result['direction']}: {result['n_files']} arquivos, "
        f"{result['rows']} linhas em {result['root']}"
    )
    print("\nAgora versione e envie:")
    print(f"  git add {result['root'].relative_to(rmcq.ROOT)}")
    print(f"  git commit -m 'exchange: pacote {result['direction']}'")
    print("  git push origin azure-exchange")
    return 0


def cmd_import_bundle(args) -> int:
    from rmcq.stages import exchange

    result = exchange.load(args.direction, overwrite=args.overwrite)
    print(
        f"\n{result['files']} arquivos importados: {result['added']} linhas novas, "
        f"{result['kept']} já existentes"
    )
    return 0


def cmd_analyze(args) -> int:
    from rmcq.stages import analyze

    result = analyze.run(args.students, args.teachers, args.depths, args.k,
                         args.datasets)
    print(f"\n{result['summary_path']}")
    print((result["summary_path"]).read_text(encoding="utf-8"))
    return 0


def cmd_status(args) -> int:
    from rmcq import env_summary
    from rmcq.backends import available_backends
    from rmcq.data import dataset_summary
    from rmcq.stages import baseline, conditions, evaluate, reflect

    print(f"rmcq {rmcq.__version__}  |  {env_summary()}\n")

    print("Escopo da rodada")
    print(f"  modelos ativos   {', '.join(ACTIVE_MODELS)}  (nos dois papéis)")
    if INACTIVE_MODELS:
        print(f"  fora da rodada   {', '.join(INACTIVE_MODELS)}")
    print(f"  pares            {len(STUDENTS) * len(TEACHERS)}  "
          f"({len(STUDENTS)} autorreflexão + "
          f"{len(STUDENTS) * len(TEACHERS) - len(STUDENTS)} externa)")
    print(f"  profundidades    {', '.join(DEPTHS)}")
    print(f"  k                {', '.join(map(str, K_VALUES))}  "
          f"(candidatos: {', '.join(map(str, K_CANDIDATES))})")
    print(f"  configs de eval  {len(STUDENTS) * len(TEACHERS) * len(DEPTHS) * len(K_VALUES)}")

    print("\nBackends")
    for name, status in available_backends().items():
        print(f"  {name:<6} {status}")

    print("\nDatasets em data/splits/")
    print(f"  {'dataset':<12}{'tipo':<12}{'treino':>8}{'valid.':>8}{'teste':>8}")
    try:
        for row in dataset_summary():
            print(f"  {row['dataset']:<12}{row['tipo']:<12}{row['train']:>8,}"
                  f"{row['validation']:>8,}{row['test']:>8,}")
    except Exception as exc:  # noqa: BLE001
        print(f"  indisponível: {exc}")

    print("\nProgresso do pipeline")
    print(f"  {'etapa':<12}{'configs':>9}{'itens':>12}{'feitos':>12}{'%':>7}")
    stages = []
    for name, fn in (
        ("baseline", lambda: baseline.plan()),
        ("reflect", lambda: reflect.plan()),
        ("eval", lambda: evaluate.plan()),
        ("retry", lambda: conditions.plan_retry()),
    ):
        try:
            rows = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<12}{'—':>9}{'—':>12}{'—':>12}{'—':>7}  ({type(exc).__name__})")
            continue
        total = sum(r.get("total", 0) for r in rows)
        done = sum(r.get("done", 0) for r in rows)
        stages.append((name, rows, total, done))
        print(f"  {name:<12}{len(rows):>9,}{total:>12,}{done:>12,}"
              f"{100 * done / max(total, 1):>6.1f}%")

    from rmcq.config import ANALYSIS_DIR, INDEX_DIR

    # Testa tamanho, não só existência: um arquivo zerado por limpeza ou por um
    # processo morto no meio da escrita existe e não serve para nada.
    def nonempty(path) -> bool:
        return path.exists() and path.stat().st_size > 0

    built = sorted(
        (p.parent.parent.name, p.parent.name)
        for p in INDEX_DIR.rglob("neighbors.npz") if nonempty(p)
    )
    print("\nÍndices de similaridade")
    if not built:
        print("  nenhum. Rode: python -m rmcq index")
    else:
        by_emb: dict[str, list[str]] = {}
        for embedder, dataset in built:
            by_emb.setdefault(embedder, []).append(dataset)
        for embedder, datasets_ in sorted(by_emb.items()):
            flag = "  <- embedder de teste, não usar no paper" if embedder == "hashing" else ""
            print(f"  {embedder:<28} {', '.join(datasets_)}{flag}")

    summary = ANALYSIS_DIR / "summary.md"
    print(f"\nAnálise: {'gerada' if nonempty(summary) else 'não gerada'}")
    return 0


def _selected_stages(args) -> list[str]:
    """Etapas a executar, respeitando --stages e --skip e preservando a ordem."""
    chosen = list(args.stages) if getattr(args, "stages", None) else list(RUN_ALL_STAGES)
    skip = set(getattr(args, "skip", None) or ())
    stages = [s for s in RUN_ALL_STAGES if s in set(chosen) and s not in skip]

    # `eval` depende de `index`; rodar sem índice quebra com erro claro, mas é
    # melhor avisar aqui do que depois de carregar um modelo de 8B.
    if "eval" in stages and "index" not in stages:
        from rmcq.config import EMBEDDER, INDEX_DIR

        slug = (getattr(args, "embedder", None) or EMBEDDER).replace("/", "_")
        if not any((INDEX_DIR / slug).rglob("neighbors.npz")):
            log.warning(
                "eval selecionado sem index, e não há índice em results/index/%s. "
                "Rode `python -m rmcq index` antes, ou inclua 'index' em --stages.", slug,
            )
    return stages


def cmd_run_all(args) -> int:
    """Executa o pipeline na ordem correta, respeitando as dependências."""
    from rmcq.config import EMBEDDER
    from rmcq.retrieval import build
    from rmcq.stages import analyze, baseline, conditions, evaluate, reflect

    runners = {
        "baseline": lambda: baseline.run(args.students, args.datasets, None,
                                         args.backend, args.limit),
        "index": lambda: build(args.datasets, embedder=args.embedder or EMBEDDER),
        "reflect": lambda: reflect.run(args.students, args.teachers, args.depths,
                                       args.datasets, args.backend, args.limit),
        "eval": lambda: evaluate.run(args.students, args.teachers, args.depths, args.k,
                                     args.datasets, args.backend,
                                     args.embedder or EMBEDDER, args.limit),
        "retry": lambda: conditions.run_retry(args.students, args.datasets,
                                              args.backend, args.limit),
        "analyze": lambda: analyze.run(args.students, args.teachers, args.depths,
                                       args.k, args.datasets),
    }

    stages = _selected_stages(args)
    skipped = [s for s in RUN_ALL_STAGES if s not in stages]

    if args.dry_run:
        print(f"etapas: {' -> '.join(stages)}")
        if skipped:
            print(f"puladas: {', '.join(skipped)}")
        per_stage = []
        for name in stages:
            if name in ESTIMATORS:
                rows, out, inn = ESTIMATORS[name](args)
                per_stage.append((name, sum(r.get("pending", 0) for r in rows), out, inn))
            else:
                # index e analyze não chamam modelo.
                per_stage.append((name, 0, 0.0, 0.0))
        _print_totals(per_stage)
        return 0

    log.info("run-all: %s", " -> ".join(stages))
    if skipped:
        log.info("puladas: %s", ", ".join(skipped))

    for name in stages:
        log.info("=" * 72)
        log.info("run-all: %s", name)
        log.info("=" * 72)
        runners[name]()
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rmcq",
        description=(
            "Framework experimental da extensão do paper de Reflection.\n"
            "Ordem do pipeline: setup-data -> notebook 01 -> setup-models -> "
            "baseline -> index -> reflect -> eval -> retry -> analyze"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMANDO")

    p = sub.add_parser("setup-data", help="baixar os datasets para data/raw/")
    _add_grid(p, students=False)
    p.add_argument("--force", action="store_true", help="rebaixar mesmo se existir")
    p.add_argument("--verify", action="store_true", help="só conferir contagens")
    p.set_defaults(func=cmd_setup_data)

    p = sub.add_parser("setup-models", help="baixar os pesos dos modelos")
    p.add_argument("--models", nargs="+", default=None, metavar="KEY",
                   help=(f"padrão: os ativos ({', '.join(ACTIVE_MODELS)}). "
                         f"'all' baixa todos: {', '.join(ALL_MODELS)}"))
    p.add_argument("--tokenizers-only", action="store_true", help="só configs e tokenizers")
    p.add_argument("--check", action="store_true", help="só checar acesso e volume")
    p.set_defaults(func=cmd_setup_models)

    p = sub.add_parser("smoke", help="carregar cada modelo e responder 1 questão")
    p.add_argument("--models", nargs="+", default=None, metavar="KEY",
                   help=f"padrão: os ativos ({', '.join(ACTIVE_MODELS)})")
    p.add_argument("--backend", choices=("vllm", "hf", "stub"), default=None)
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("baseline", help="etapa 1: responder tudo sem reflexão")
    _add_grid(p, splits=True)
    _add_common(p)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("index", help="etapa 3: índice de similaridade entre questões")
    _add_grid(p, students=False)
    p.add_argument("--embedder", default=None, help="padrão: RMCQ_EMBEDDER do .env")
    p.add_argument("--max-k", type=_pos_int, default=None, help=f"padrão: max({K_VALUES})")
    p.add_argument("--force", action="store_true", help="recalcular")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("reflect", help="etapa 2: professor escreve as reflexões")
    _add_grid(p, teachers=True, depths=True)
    _add_common(p)
    p.set_defaults(func=cmd_reflect)

    p = sub.add_parser("eval", help="etapa 4: responder o teste com reflexões recuperadas")
    _add_grid(p, teachers=True, depths=True, ks=True)
    _add_common(p)
    p.add_argument("--embedder", default=None)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("retry", help="controle: retry com feedback, sem reflexão")
    _add_grid(p)
    _add_common(p)
    p.set_defaults(func=cmd_retry)

    p = sub.add_parser(
        "export-bundle",
        help="empacotar dados para trocar com o ambiente Azure (via git)",
    )
    p.add_argument("--direction", required=True, choices=list(EXCHANGE_DIRECTIONS),
                   help=("to-azure: perguntas + respostas dos alunos. "
                         "from-azure: reflexões geradas pelos professores de API"))
    _add_grid(p, teachers=True, depths=True)
    p.set_defaults(func=cmd_export_bundle)

    p = sub.add_parser(
        "import-bundle",
        help="conferir e materializar um pacote de troca recebido por git",
    )
    p.add_argument("--direction", required=True, choices=list(EXCHANGE_DIRECTIONS))
    p.add_argument("--overwrite", action="store_true",
                   help="substituir os arquivos locais em vez de juntar por uid")
    p.set_defaults(func=cmd_import_bundle)

    p = sub.add_parser("analyze", help="etapa 5: métricas da seção 5 do Caderno")
    _add_grid(p, teachers=True, depths=True, ks=True)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("status", help="o que já foi feito e o que falta")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("run-all", help="pipeline inteiro, na ordem")
    _add_grid(p, teachers=True, depths=True, ks=True)
    _add_common(p)
    p.add_argument("--embedder", default=None)
    p.add_argument("--stages", nargs="+", default=None, choices=list(RUN_ALL_STAGES),
                   metavar="ETAPA",
                   help=f"quais rodar (padrão: todas — {', '.join(RUN_ALL_STAGES)})")
    p.add_argument("--skip", nargs="+", default=None, choices=list(RUN_ALL_STAGES),
                   metavar="ETAPA", help="quais pular, ex: --skip retry")
    p.set_defaults(func=cmd_run_all)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    ensure_dirs()
    if getattr(args, "log_file", False):
        path = log_to_file(args.command)
        log.info("log espelhado em %s", path)

    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        # Cada etapa grava incrementalmente, então interromper é seguro: rodar
        # de novo retoma de onde parou.
        log.warning("interrompido. O progresso gravado está preservado; rode de novo para retomar.")
        return 130
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    except (ValueError, KeyError) as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
