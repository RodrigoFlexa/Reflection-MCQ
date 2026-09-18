"""Recortar uma passada do professor não pode inventar linha nem apagar lacuna.

A rodada final do GPT-5.4 é um recorte deliberado: ele ensina só os alunos
pequenos, e o RACE saiu do estudo. As duas coisas precisam acontecer como
ausência de trabalho — a célula continua cobrada em gaps.json — e nunca como
linha `not_generated`, que afirma que a condição foi tentada e falhou.
"""
import json
import sys

import pytest

import run_experiment as run
from rmcq import grid

SEM_RACE = [d for d in run.DEFAULT_DATASETS if d != "race"]
ALUNOS = ["phi2", "llama3.2-3b", "qwen2.5-3b"]


def item(dataset, split, index):
    return dict(uid=f"{dataset}-{split}-{index}", dataset=dataset, split=split,
                problem_type="process", context="A passage about " + split,
                question=f"Which option {index}?", choices=[{"label": c, "text": c} for c in "ABCD"],
                answerKey="C", num_choices=4, rationale=None,
                article_uid=f"article-{split}", race_subset=None)


@pytest.fixture
def bancada(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "find_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    for dataset in run.DEFAULT_DATASETS:
        folder = tmp_path / "data/processed" / dataset
        run.save_jsonl(folder / "train.jsonl", [item(dataset, "train", 0)])
        run.save_jsonl(folder / "validation.jsonl", [item(dataset, "validation", 1)])

    def retrieval(state, model, device):
        return [{"dataset": d, "val_uid": v["uid"], "eval_uid": v["uid"], "eval_split": "validation",
                 "source_uid": data["train"][0]["uid"], "similarity": .9,
                 "validation_item": v, "source_item": data["train"][0]}
                for d, data in state.items() for v in data["validation"]]
    monkeypatch.setattr(run, "retrieve_top1", retrieval)
    return tmp_path


@pytest.fixture
def rodada(bancada, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_experiment.py", "prepare", "--preset", "validation-threshold",
        "--backend", "stub", "--models", ",".join(ALUNOS)])
    run.main()
    rid = json.loads(next((bancada / "experiment_exchange").glob("*/manifest.json"))
                     .read_text(encoding="utf-8"))["experiment_id"]

    def stage(nome, *extra):
        monkeypatch.setattr(sys, "argv", ["run_experiment.py", nome, "--experiment-id", rid,
                                          "--backend", "stub", "--teacher-backend", "stub", *extra])
        run.main()

    return bancada, rid, stage


def test_recorte_de_alunos_e_datasets_deixa_o_resto_pendente(rodada):
    bancada, rid, stage = rodada
    stage("teacher", "--only-teachers", grid.EXTERNAL_TEACHER,
          "--only-students", "llama3.2-3b", "--only-datasets", ",".join(SEM_RACE))

    exchange = bancada / "experiment_exchange" / rid
    recibo = json.loads((exchange / "teacher_receipt.json").read_text(encoding="utf-8"))
    assert recibo["students_this_pass"] == ["llama3.2-3b"]
    assert recibo["datasets_this_pass"] == SEM_RACE
    assert recibo["taught"][grid.EXTERNAL_TEACHER] == ["llama3.2-3b"]
    assert recibo["complete"] is False
    # Os alunos que ficaram de fora continuam sendo cobrados, um por um.
    assert f"{grid.EXTERNAL_TEACHER}/phi2" in recibo["missing_pairs"]
    assert f"{grid.EXTERNAL_TEACHER}/qwen2.5-3b" in recibo["missing_pairs"]

    reflexoes = run.load_teacher_reflections(exchange, grid.EXTERNAL_TEACHER, "llama3.2-3b")
    assert {r["dataset"] for r in reflexoes} == set(SEM_RACE), "o RACE não podia ter sido refletido"
    assert not (exchange / "teacher" / grid.EXTERNAL_TEACHER /
                "student_reflections" / "phi2.jsonl").exists()


def test_dataset_fora_da_passada_nao_vira_linha_no_finish(rodada):
    bancada, rid, stage = rodada
    stage("self-eval")
    # Um professor aberto já tinha refletido sobre TUDO, RACE incluso.
    stage("teacher", "--only-teachers", "llama3.1-8b")
    stage("teacher", "--only-teachers", grid.EXTERNAL_TEACHER,
          "--only-students", "llama3.2-3b", "--only-datasets", ",".join(SEM_RACE))
    stage("finish", "--only-datasets", ",".join(SEM_RACE))

    outcomes = run.load_jsonl(bancada / "data/results/reflection_top1" / rid /
                              "analysis/all_outcomes.jsonl")
    professor = [r for r in outcomes if r["condition"].startswith("teacher")]
    externo = [r for r in professor if r["teacher_model"] == grid.EXTERNAL_TEACHER]
    assert externo, "o braço do professor tinha de existir para o aluno pedido"
    assert {r["model"] for r in externo} == {"llama3.2-3b"}
    assert {r["dataset"] for r in externo} == set(SEM_RACE)
    assert not [r for r in externo if r["dataset"] == "race"], \
        "RACE fora da passada não pode virar linha do professor"
    # O recorte não apaga trabalho feito: o professor aberto mantém o RACE.
    aberto = [r for r in professor if r["teacher_model"] == "llama3.1-8b"]
    assert {r["dataset"] for r in aberto} == set(run.DEFAULT_DATASETS)
    # Nenhuma linha inventada: o que não foi gerado simplesmente não está lá.
    assert not [r for r in professor if r["finish_reason"] == "not_generated"]
    # E a autorreflexão continua cobrindo todos os datasets, inclusive o RACE.
    baseline = [r for r in outcomes if r["condition"] == "baseline" and r["model"] == "llama3.2-3b"]
    assert {r["dataset"] for r in baseline} == set(run.DEFAULT_DATASETS)

    recibo = json.loads((bancada / "data/results/reflection_top1" / rid /
                         "finish_receipt.json").read_text(encoding="utf-8"))
    assert recibo["datasets_with_teacher"] == sorted(SEM_RACE)

    # Repetir o mesmo comando é barato: o recorte já está satisfeito em disco.
    antes = len(outcomes)
    chamadas = []
    original = run.cached_generate
    try:
        run.cached_generate = lambda *a, **k: (chamadas.append(a[1]), original(*a, **k))[1]
        stage("finish", "--only-datasets", ",".join(SEM_RACE))
    finally:
        run.cached_generate = original
    depois = run.load_jsonl(bancada / "data/results/reflection_top1" / rid /
                            "analysis/all_outcomes.jsonl")
    assert len(depois) == antes
    assert not [p for p in chamadas if "llama3.2-3b" in str(p)], \
        "o aluno já completo não podia ser gerado de novo"


def test_recorte_so_vale_onde_faz_sentido(rodada, monkeypatch):
    bancada, rid, stage = rodada
    with pytest.raises(SystemExit):
        stage("finish", "--only-students", "phi2")
    with pytest.raises(SystemExit):
        stage("self-eval", "--only-datasets", "arc")
    with pytest.raises(ValueError, match="fora desta run"):
        stage("teacher", "--only-teachers", grid.EXTERNAL_TEACHER, "--only-datasets", "gsm8k")
    with pytest.raises(ValueError, match="fora desta run"):
        stage("teacher", "--only-teachers", grid.EXTERNAL_TEACHER, "--only-students", "ministral-3-8b")
