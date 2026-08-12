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
    SELFCONS_N,
    STUDENTS,
    TEACHERS,
    ensure_dirs,
)
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
    "selfcons": {"in": 180, "out": 260},
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


def _pos_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("precisa ser positivo")
    return n


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dry-run", action="store_true", help="estimar sem executar")
    p.add_argument("--backend", choices=("vllm", "hf", "stub"), default=None,
                   help="sobrepõe RMCQ_BACKEND do .env")
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
                       help="padrão: todos os splits disponíveis")


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

    rows = baseline.plan(args.students, args.datasets, args.splits)
    if args.dry_run:
        pending = sum(r["pending"] for r in rows)
        _print_plan(rows, pending * EST_TOKENS["baseline"]["out"],
                    pending * EST_TOKENS["baseline"]["in"])
        return 0
    baseline.run(args.students, args.datasets, args.splits, args.backend, args.limit)
    return 0


def cmd_reflect(args) -> int:
    from rmcq.stages import reflect

    rows = reflect.plan(args.students, args.teachers, args.depths, args.datasets)
    if args.dry_run:
        est = EST_TOKENS["reflect"]
        out = sum(r["pending"] * est[r["depth"]]["out"] for r in rows)
        inn = sum(r["pending"] * est[r["depth"]]["in"] for r in rows)
        _print_plan(rows, out, inn)
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

    rows = evaluate.plan(args.students, args.teachers, args.depths, args.k, args.datasets)
    if args.dry_run:
        est = EST_TOKENS["eval"]
        out = sum(r["pending"] * est["in_per_reflection"]["out"] for r in rows)
        inn = sum(
            r["pending"] * (
                est["in_base"] + r["k"] * (
                    est["in_per_reflection"]["simple"] if r["depth"] == "simple"
                    else est["in_per_reflection_complex"]
                )
            )
            for r in rows
        )
        _print_plan(rows, out, inn)
        return 0
    evaluate.run(args.students, args.teachers, args.depths, args.k, args.datasets,
                 args.backend, args.embedder or EMBEDDER, args.limit)
    return 0


def cmd_retry(args) -> int:
    from rmcq.stages import conditions

    rows = conditions.plan_retry(args.students, args.datasets)
    if args.dry_run:
        pending = sum(r["pending"] for r in rows)
        _print_plan(rows, pending * EST_TOKENS["retry"]["out"],
                    pending * EST_TOKENS["retry"]["in"])
        return 0
    conditions.run_retry(args.students, args.datasets, args.backend, args.limit)
    return 0


def cmd_selfcons(args) -> int:
    from rmcq.stages import conditions

    rows = conditions.plan_selfcons(args.students, args.datasets, args.n)
    if args.dry_run:
        gens = sum(r["generations"] for r in rows)
        _print_plan(rows, gens * EST_TOKENS["selfcons"]["out"],
                    sum(r["pending"] for r in rows) * EST_TOKENS["selfcons"]["in"])
        return 0
    conditions.run_selfcons(args.students, args.datasets, args.n, args.backend, args.limit)
    return 0


def cmd_analyze(args) -> int:
    from rmcq.stages import analyze

    result = analyze.run(args.students, args.teachers, args.depths, args.k,
                         args.datasets, args.n)
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
        ("selfcons", lambda: conditions.plan_selfcons()),
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


def cmd_run_all(args) -> int:
    """Executa o pipeline inteiro na ordem correta, respeitando as dependências."""
    from rmcq.config import EMBEDDER
    from rmcq.retrieval import build
    from rmcq.stages import analyze, baseline, conditions, evaluate, reflect

    steps = [
        ("baseline", lambda: baseline.run(args.students, args.datasets, None,
                                          args.backend, args.limit)),
        ("index", lambda: build(args.datasets, embedder=args.embedder or EMBEDDER)),
        ("reflect", lambda: reflect.run(args.students, args.teachers, args.depths,
                                        args.datasets, args.backend, args.limit)),
        ("eval", lambda: evaluate.run(args.students, args.teachers, args.depths, args.k,
                                      args.datasets, args.backend,
                                      args.embedder or EMBEDDER, args.limit)),
        ("retry", lambda: conditions.run_retry(args.students, args.datasets,
                                               args.backend, args.limit)),
        ("selfcons", lambda: conditions.run_selfcons(args.students, args.datasets,
                                                     args.n, args.backend, args.limit)),
        ("analyze", lambda: analyze.run(args.students, args.teachers, args.depths,
                                        args.k, args.datasets, args.n)),
    ]

    if args.dry_run:
        print("run-all executaria, nesta ordem:")
        for name, _ in steps:
            print(f"  {name}")
        print("\nUse --dry-run em cada subcomando para estimar o custo individual.")
        return 0

    for name, fn in steps:
        log.info("=" * 72)
        log.info("run-all: %s", name)
        log.info("=" * 72)
        fn()
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
            "baseline -> index -> reflect -> eval -> retry/selfcons -> analyze"
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

    p = sub.add_parser("selfcons", help="controle: self-consistency com voto majoritário")
    _add_grid(p)
    _add_common(p)
    p.add_argument("-n", type=_pos_int, default=SELFCONS_N,
                   help=f"amostras por questão (padrão {SELFCONS_N})")
    p.set_defaults(func=cmd_selfcons)

    p = sub.add_parser("analyze", help="etapa 5: métricas da seção 5 do Caderno")
    _add_grid(p, teachers=True, depths=True, ks=True)
    p.add_argument("-n", type=_pos_int, default=SELFCONS_N,
                   help="N de self-consistency a considerar")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("status", help="o que já foi feito e o que falta")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("run-all", help="pipeline inteiro, na ordem")
    _add_grid(p, teachers=True, depths=True, ks=True)
    _add_common(p)
    p.add_argument("--embedder", default=None)
    p.add_argument("-n", type=_pos_int, default=SELFCONS_N)
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
