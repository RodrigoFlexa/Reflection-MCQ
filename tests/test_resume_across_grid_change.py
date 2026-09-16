"""Trocar um aluno da grade não pode custar a regeneração dos outros.

O id do run é o hash da configuração, então mudar a lista de alunos cria um run
novo. Sem adoção, esse run novo regeraria do zero os oito alunos que não
mudaram — dias de GPU para reproduzir arquivos idênticos aos que já estão no
disco. Este teste é o que garante que isso não volta a acontecer.
"""
import json
import sys

import pytest

import run_experiment as run


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


def prepara(models, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_experiment.py", "prepare", "--preset", "validation-threshold",
        "--backend", "stub", "--models", ",".join(models)])
    run.main()


def gerados(chamadas, prefixo):
    return sorted({p.parent.name for p in chamadas if p.name.startswith(prefixo)})


def test_troca_de_aluno_nao_regenera_os_demais(bancada, monkeypatch):
    antigos = ["phi2", "llama3.1-8b", "qwen2.5-3b"]
    prepara(antigos, monkeypatch)
    primeiro = json.loads(next((bancada / "experiment_exchange").glob("*/manifest.json"))
                          .read_text(encoding="utf-8"))["experiment_id"]

    chamadas = []
    original = run.cached_generate
    monkeypatch.setattr(run, "cached_generate",
                        lambda *a, **k: (chamadas.append(a[1]), original(*a, **k))[1])

    # Um aluno sai, outro entra: configuração nova, id novo, mesmos pares.
    novos = ["phi2", "llama3.1-8b", "deepseek-r1-distill-qwen-1.5b"]
    prepara(novos, monkeypatch)
    ids = {json.loads(p.read_text(encoding="utf-8"))["experiment_id"]
           for p in (bancada / "experiment_exchange").glob("*/manifest.json")}
    segundo = next(i for i in ids if i != primeiro)
    assert len(ids) == 2, "a troca de aluno tem de criar um run novo"

    # Só o aluno que entrou gerou respostas de treino.
    assert gerados(chamadas, "train_answers") == ["deepseek-r1-distill-qwen-1.5b"]
    assert gerados(chamadas, "self_") == ["deepseek-r1-distill-qwen-1.5b"]

    # E, ainda assim, o run novo está completo e é autocontido.
    exchange = bancada / "experiment_exchange" / segundo
    for model in novos:
        treino = exchange / "students" / model / "train.jsonl"
        assert treino.exists(), f"{model} ficou sem tentativas de treino"
        assert run.load_jsonl(treino), f"{model} ficou com treino vazio"
    recibo = json.loads((exchange / "prepare_receipt.json").read_text(encoding="utf-8"))
    assert sorted(recibo["student_models"]) == sorted(novos)
    assert recibo["complete"] is True

    # O run anterior continua íntegro: adoção copia, não move.
    anterior = bancada / "experiment_exchange" / primeiro
    assert all((anterior / "students" / m / "train.jsonl").exists() for m in antigos)


def test_adocao_recusa_configuracao_de_geracao_diferente():
    base = {"pipeline_version": "v5", "answer_prompt": "A", "student_reflection_prompts": {},
            "transfer_prompt": "T", "generation_policy": {}, "generation_profile": "final",
            "phi2_stop_sequences": [], "judge_model": "llama3.1-8b",
            "embedding_model": "bge", "datasets": ["arc"], "eval_split": "validation",
            "validation_cap": None, "train_cap": None, "seed": 42,
            "model_specs": {"phi2": {"repo_id": "microsoft/phi-2", "extra_kwargs": {}}},
            "reflection_max_tokens": {"phi2": {"simple": 512}}}
    assert run.adoptable_students(base, base, ["phi2"]) == ["phi2"]

    # Prompt diferente invalida tudo: o número não mediria a mesma coisa.
    outro_prompt = {**base, "transfer_prompt": "outro"}
    assert run.adoptable_students(outro_prompt, base, ["phi2"]) == []

    # Orçamento diferente invalida só o modelo afetado.
    outro_budget = {**base, "reflection_max_tokens": {"phi2": {"simple": 256}}}
    assert run.adoptable_students(outro_budget, base, ["phi2"]) == []

    # Pesos diferentes sob a mesma chave também.
    outro_peso = {**base, "model_specs": {"phi2": {"repo_id": "outro", "extra_kwargs": {}}}}
    assert run.adoptable_students(outro_peso, base, ["phi2"]) == []


def test_professores_podem_ser_divididos_entre_maquinas(bancada, monkeypatch):
    """Quatro professores precisam de GPU; o GPT-5.4 precisa de credencial Azure.

    Cada máquina roda o subconjunto que consegue, e o recibo só fecha quando os
    dois lados tiverem passado. Nenhum dos dois enxerga a passada do outro a não
    ser pelo que ficou no disco.
    """
    from rmcq import grid
    prepara(list(grid.STUDENTS), monkeypatch)
    rid = json.loads(next((bancada / "experiment_exchange").glob("*/manifest.json"))
                     .read_text(encoding="utf-8"))["experiment_id"]
    recibo = bancada / "experiment_exchange" / rid / "teacher_receipt.json"

    def teacher(only=None):
        argv = ["run_experiment.py", "teacher", "--experiment-id", rid,
                "--teacher-backend", "stub", "--backend", "stub"]
        monkeypatch.setattr(sys, "argv", argv + (["--only-teachers", only] if only else []))
        run.main()
        return json.loads(recibo.read_text(encoding="utf-8"))

    # Máquina 1: só o professor externo.
    estado = teacher(grid.EXTERNAL_TEACHER)
    assert estado["complete"] is False, "com quatro professores pendentes o recibo não fecha"
    assert estado["generated_this_pass"] == [grid.EXTERNAL_TEACHER]
    assert estado["taught"][grid.EXTERNAL_TEACHER] == sorted(grid.STUDENTS)
    assert all(estado["taught"][t] == [] for t in grid.OPEN_TEACHERS)

    # Máquina 2: os quatro abertos. O recibo soma as duas passadas.
    estado = teacher(",".join(grid.OPEN_TEACHERS))
    assert estado["missing_pairs"] == []
    assert estado["complete"] is True
    assert {t: estado["taught"][t] for t in grid.TEACHERS} == {
        t: sorted(grid.students_for(t)) for t in grid.TEACHERS}

    # Um professor fora da grade é erro, não trabalho silenciosamente ignorado.
    with pytest.raises(ValueError, match="fora da grade"):
        teacher("phi2")


def test_padronizar_a_destilacao_regenera_so_o_deepseek(bancada, monkeypatch):
    """Trocar a base do DeepSeek custa o DeepSeek, e só ele.

    A grade passou a exigir a destilação sobre Llama. O run anterior rodou a
    destilação sobre Qwen3 sob outro nome: são pesos diferentes, então aquelas
    gerações não valem e o modelo tem de ser refeito. Os outros oito não têm
    nada a ver com isso e não podem ser regerados junto.
    """
    from rmcq import grid
    antigos = ["phi2", "llama3.1-8b", "deepseek-r1-0528-qwen3-8b"]
    prepara(antigos, monkeypatch)

    chamadas = []
    original = run.cached_generate
    monkeypatch.setattr(run, "cached_generate",
                        lambda *a, **k: (chamadas.append(a[1]), original(*a, **k))[1])

    prepara(["phi2", "llama3.1-8b", "deepseek-r1-8b"], monkeypatch)

    assert gerados(chamadas, "train_answers") == ["deepseek-r1-8b"]
    assert grid.canonical("deepseek-r1-0528-qwen3-8b") == "deepseek-r1-0528-qwen3-8b", \
        "o modelo aposentado não pode voltar a contar como o DeepSeek da grade"


def test_pesos_diferentes_nunca_sao_adotados_pelo_nome():
    """O apelido unifica o rótulo, nunca a geração.

    Se a adoção olhasse só para o nome, os artefatos da destilação Qwen3
    entrariam como se fossem da Llama e ninguém veria a troca no número.
    """
    from rmcq.config import MODELS
    base = {"pipeline_version": "v5", "answer_prompt": "A", "student_reflection_prompts": {},
            "transfer_prompt": "T", "generation_policy": {}, "generation_profile": "final",
            "phi2_stop_sequences": [], "judge_model": "llama3.1-8b", "embedding_model": "bge",
            "datasets": ["arc"], "eval_split": "validation", "validation_cap": None,
            "train_cap": None, "seed": 42, "reflection_max_tokens": {}}

    def spec(key):
        return {"repo_id": MODELS[key].repo_id, "extra_kwargs": MODELS[key].extra_kwargs}

    # Nome antigo, mesmos pesos: adota.
    anterior = {**base, "model_specs": {"deepseek-r1-distill-llama-8b": spec("deepseek-r1-distill-llama-8b")}}
    atual = {**base, "model_specs": {"deepseek-r1-8b": spec("deepseek-r1-8b")}}
    assert run.adoptable_students(anterior, atual, ["deepseek-r1-8b"]) == ["deepseek-r1-8b"]

    # Outra destilação: recusa, mesmo que o rótulo fosse o mesmo.
    outra_base = {**base, "model_specs": {"deepseek-r1-8b": spec("deepseek-r1-0528-qwen3-8b")}}
    assert run.adoptable_students(outra_base, atual, ["deepseek-r1-8b"]) == []


def test_finish_sem_o_professor_externo_nao_inventa_linha(bancada, monkeypatch):
    """Rodar o que dá hoje não pode sujar o que falta para amanhã.

    Sem credencial Azure, os quatro professores locais rodam e o GPT-5.4 não.
    O `finish` tem de gerar os braços dos quatro e simplesmente NÃO escrever os
    do quinto — se escrevesse linhas `not_generated`, o arquivo passaria a
    afirmar que a condição foi tentada e falhou, e a lacuna sumiria do
    gaps.json como se o trabalho estivesse feito.
    """
    from rmcq import grid
    prepara(list(grid.STUDENTS), monkeypatch)
    rid = json.loads(next((bancada / "experiment_exchange").glob("*/manifest.json"))
                     .read_text(encoding="utf-8"))["experiment_id"]

    def stage(nome, *extra):
        monkeypatch.setattr(sys, "argv", ["run_experiment.py", nome, "--experiment-id", rid,
                                          "--backend", "stub", "--teacher-backend", "stub", *extra])
        run.main()

    stage("self-eval")
    stage("teacher", "--only-teachers", ",".join(grid.OPEN_TEACHERS))
    stage("finish")

    outcomes = run.load_jsonl(bancada / "data/results/reflection_top1" / rid /
                              "analysis/all_outcomes.jsonl")
    professores = {r.get("teacher_model") for r in outcomes if r["condition"].startswith("teacher")}
    assert professores == set(grid.OPEN_TEACHERS), "o professor que não rodou não pode ter linhas"
    assert grid.EXTERNAL_TEACHER not in professores

    # Nenhuma linha inventada: tudo que existe foi de fato gerado.
    assert not [r for r in outcomes if r.get("teacher_model") == grid.EXTERNAL_TEACHER]
    recibo = json.loads((bancada / "data/results/reflection_top1" / rid /
                         "finish_receipt.json").read_text(encoding="utf-8"))
    assert recibo["teacher_arms_pending"] == sorted(
        f"{grid.EXTERNAL_TEACHER}/{m}" for m in grid.STUDENTS)

    # E quando o professor externo finalmente rodar, o mesmo comando completa.
    stage("teacher", "--only-teachers", grid.EXTERNAL_TEACHER)
    stage("finish")
    outcomes = run.load_jsonl(bancada / "data/results/reflection_top1" / rid /
                              "analysis/all_outcomes.jsonl")
    assert {r.get("teacher_model") for r in outcomes if r["condition"].startswith("teacher")} == \
        set(grid.TEACHERS)


def test_adocao_evita_reavaliar_e_nao_so_retreinar(bancada, monkeypatch):
    """Adotar o treino e reavaliar tudo não seria retomada, seria refazer.

    A avaliação é uma geração por questão por condição — é onde o tempo de GPU
    vai. Se ela não fosse adotada junto, trocar um aluno da grade custaria
    reavaliar os outros oito, que é justamente o que a adoção existe para
    evitar.
    """
    antigos = ["phi2", "llama3.1-8b", "qwen2.5-3b"]
    prepara(antigos, monkeypatch)
    primeiro = json.loads(next((bancada / "experiment_exchange").glob("*/manifest.json"))
                          .read_text(encoding="utf-8"))["experiment_id"]
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "self-eval",
                                      "--experiment-id", primeiro, "--backend", "stub"])
    run.main()

    chamadas = []
    original = run.cached_generate
    monkeypatch.setattr(run, "cached_generate",
                        lambda *a, **k: (chamadas.append(a[1]), original(*a, **k))[1])

    novos = ["phi2", "llama3.1-8b", "deepseek-r1-1.5b"]
    prepara(novos, monkeypatch)
    segundo = next(json.loads(p.read_text(encoding="utf-8"))["experiment_id"]
                   for p in (bancada / "experiment_exchange").glob("*/manifest.json")
                   if json.loads(p.read_text(encoding="utf-8"))["experiment_id"] != primeiro)
    chamadas.clear()
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "self-eval",
                                      "--experiment-id", segundo, "--backend", "stub"])
    run.main()

    # Só o aluno novo é avaliado; os dois adotados já têm as respostas em disco.
    avaliados = gerados(chamadas, "validation")
    assert avaliados == ["deepseek-r1-1.5b"], f"reavaliou demais: {avaliados}"

    # E o run novo contém os três, com o nome que a grade usa hoje.
    outcomes = run.load_jsonl(bancada / "data/results/reflection_top1" / segundo /
                              "self_eval/analysis/all_outcomes.jsonl")
    assert {r["model"] for r in outcomes} == set(novos)
    adotado = [r for r in outcomes if r["model"] == "deepseek-r1-1.5b"]
    assert adotado, "o aluno novo tem de estar avaliado"


def test_finish_nao_rele_reflexoes_a_cada_par(bancada, monkeypatch):
    """Saber se um professor rodou não pode custar ler o arquivo dele.

    A pergunta é feita uma vez por par de avaliação. Numa versão anterior ela
    abria e parseava o arquivo inteiro de reflexões a cada chamada — dezenas de
    milhares de leituras do mesmo arquivo, que travaram o estágio a 100% de CPU
    sem gerar uma única resposta.
    """
    from rmcq import grid
    prepara(list(grid.STUDENTS), monkeypatch)
    rid = json.loads(next((bancada / "experiment_exchange").glob("*/manifest.json"))
                     .read_text(encoding="utf-8"))["experiment_id"]

    def stage(nome, *extra):
        monkeypatch.setattr(sys, "argv", ["run_experiment.py", nome, "--experiment-id", rid,
                                          "--backend", "stub", "--teacher-backend", "stub", *extra])
        run.main()

    stage("self-eval")
    stage("teacher", "--only-teachers", ",".join(grid.OPEN_TEACHERS))

    leituras = []
    original = run.load_teacher_reflections
    monkeypatch.setattr(run, "load_teacher_reflections",
                        lambda ex, t, s: (leituras.append((t, s)), original(ex, t, s))[1])
    stage("finish")

    # No máximo uma leitura por par professor-aluno de fato ensinado.
    pares = len([1 for t in grid.OPEN_TEACHERS for s in grid.students_for(t)])
    assert len(leituras) <= pares, (
        f"{len(leituras)} leituras para {pares} pares: a checagem voltou ao laço de pares")
