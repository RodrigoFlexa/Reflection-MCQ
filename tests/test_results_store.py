"""A grade final e o armazém canônico: o que não pode regredir."""
import json

import pytest

from rmcq import grid
from rmcq import results_store as store
from rmcq.results_store import Source


def test_grade_final_fecha_como_o_protocolo_manda():
    grid.validate()
    assert len(grid.STUDENTS) == 9
    assert len(grid.SMALL_STUDENTS) == 5
    # O professor externo ensina todo mundo; os abertos, só os de até 3B.
    assert grid.students_for(grid.EXTERNAL_TEACHER) == grid.STUDENTS
    for teacher in grid.OPEN_TEACHERS:
        assert grid.students_for(teacher) == grid.SMALL_STUDENTS
        assert teacher not in grid.SMALL_STUDENTS
    # Nenhum 8B ensina um 8B, que é a regra que motivou a grade.
    for teacher, student in grid.teacher_pairs():
        if teacher in grid.OPEN_TEACHERS:
            assert student in grid.SMALL_STUDENTS
    assert len(grid.teacher_pairs()) == 29
    # Aluno grande: três células próprias mais as duas do professor externo.
    assert len(grid.conditions_for("llama3.1-8b")) == 5
    # O DeepSeek é padronizado na destilação sobre Llama.
    assert grid.canonical("deepseek-r1-distill-llama-8b") == "deepseek-r1-8b"
    assert grid.canonical("deepseek-r1-distill-qwen-1.5b") == "deepseek-r1-1.5b"
    assert grid.canonical("phi2") == "phi2"
    # O 0528 é outra base: aposentado, e não pode virar o DeepSeek da grade.
    assert grid.canonical("deepseek-r1-0528-qwen3-8b") == "deepseek-r1-0528-qwen3-8b"
    assert "deepseek-r1-0528-qwen3-8b" in grid.RETIRED_STUDENTS
    # Aluno pequeno: as três próprias mais duas por cada um dos cinco professores.
    assert len(grid.conditions_for("phi2")) == 13
    assert len(grid.expected_cells()) == 85


def escreve(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def linha(**kwargs):
    base = dict(model="phi2", dataset="arc", val_uid="arc-0", condition="baseline",
                similarity=.9, correct=1, eval_split="validation")
    return {**base, **kwargs}


def test_normalize_torna_o_professor_explicito_sem_inventar_dado():
    antiga = linha(condition="teacher_simple")
    assert "teacher_model" not in antiga
    # Quando só havia um professor, a ausência do campo afirmava o externo.
    assert store.normalize(antiga, "validation")["teacher_model"] == grid.EXTERNAL_TEACHER
    # Uma condição sem professor nunca ganha um.
    assert store.normalize(linha(), "validation")["teacher_model"] is None
    # O split vem do arquivo só quando a linha não o declara.
    assert store.normalize(linha(eval_split=None), "validation")["eval_split"] == "validation"
    assert store.normalize(linha(eval_split="test"), "test")["eval_split"] == "test"


def test_a_primeira_fonte_vence_e_professores_diferentes_convivem(tmp_path):
    boa = escreve(tmp_path / "boa.jsonl", [linha(correct=1)])
    velha = escreve(tmp_path / "velha.jsonl", [
        linha(correct=0),                                    # mesma célula: duplicata
        linha(condition="teacher_simple", teacher_model="llama3.1-8b", correct=1),
        linha(condition="teacher_simple", teacher_model="qwen2.5-7b", correct=0),
    ])
    report = store.merge(tmp_path, "validation", [
        Source(path=boa, stamp={"source_run": "novo"}),
        Source(path=velha, stamp={"source_run": "velho"}),
    ])
    assert report.duplicates_skipped == 1
    rows = list(store.stream_outcomes(tmp_path, "validation"))
    assert len(rows) == 3
    baseline = next(r for r in rows if r["condition"] == "baseline")
    assert baseline["correct"] == 1 and baseline["source_run"] == "novo"
    # Dois professores na mesma questão são duas medidas, não uma colisão.
    assert {r["teacher_model"] for r in rows if r["condition"] == "teacher_simple"} == {
        "llama3.1-8b", "qwen2.5-7b"}


def test_merge_recusa_misturar_splits(tmp_path):
    fonte = escreve(tmp_path / "teste.jsonl", [linha(eval_split="test")])
    with pytest.raises(ValueError, match="split"):
        store.merge(tmp_path, "validation", [Source(path=fonte)])


def test_lacunas_separam_celula_ausente_de_celula_pela_metade(tmp_path):
    rows = []
    for uid in ("arc-0", "arc-1"):
        for condition in grid.SELF_CONDITIONS:
            rows.append(linha(val_uid=uid, condition=condition))
    # Um professor que cobriu só metade das questões: retomada, não trabalho novo.
    rows.append(linha(val_uid="arc-0", condition="teacher_simple", teacher_model="llama3.1-8b"))
    escreve(tmp_path / "f.jsonl", rows)
    store.merge(tmp_path, "validation", [Source(path=tmp_path / "f.jsonl")])
    table, universe, provenance = store.coverage(store.stream_outcomes(tmp_path, "validation"))
    holes = store.gaps(table, universe)
    assert universe["arc"] == {"arc-0", "arc-1"}

    def achar(entries, condition, teacher):
        return [e for e in entries if e["condition"] == condition and e["teacher_model"] == teacher]

    assert achar(holes["cells_partial"], "teacher_simple", "llama3.1-8b")[0]["rows_missing"] == 1
    assert achar(holes["cells_missing"], "teacher_simple", "qwen2.5-7b")[0]["rows_missing"] == 2
    # phi2 é aluno pequeno: os cinco professores são cobrados dele.
    esperados = {t for c, t in grid.conditions_for("phi2") if t}
    cobrados = {e["teacher_model"] for e in holes["cells_missing"] + holes["cells_partial"]
                if e["model"] == "phi2" and e["teacher_model"]}
    assert cobrados == esperados


def test_provenancia_mostra_revisao_e_checkpoint(tmp_path):
    escreve(tmp_path / "v5.jsonl", [linha()])
    escreve(tmp_path / "v4.jsonl", [linha(model="deepseek-r1-distill-llama-8b")])
    store.merge(tmp_path, "validation", [
        Source(path=tmp_path / "v5.jsonl", stamp={"source_run": "a", "pipeline_version": "v5"}),
        Source(path=tmp_path / "v4.jsonl", stamp={"source_run": "b", "pipeline_version": "v4"}),
    ])
    _, _, provenance = store.coverage(store.stream_outcomes(tmp_path, "validation"))
    # O nome histórico vira o canônico, mas o checkpoint de origem sobrevive.
    assert {(p["model"], p["checkpoint"], p["pipeline_version"]) for p in provenance} == {
        ("phi2", "phi2", "v5"),
        ("deepseek-r1-8b", "deepseek-r1-distill-llama-8b", "v4")}


def test_nome_historico_da_destilacao_llama_conta_como_o_deepseek(tmp_path):
    """`deepseek-r1-distill-llama-8b` e `deepseek-r1-8b` são o mesmo modelo.

    É o mesmo checkpoint sob dois nomes, então as linhas antigas continuam
    valendo e um recorte escrito com qualquer um dos nomes casa com as duas.
    """
    fonte = escreve(tmp_path / "f.jsonl", [linha(model="deepseek-r1-distill-llama-8b")])
    store.merge(tmp_path, "validation", [Source(path=fonte, stamp={"source_run": "r"})])
    rows = list(store.stream_outcomes(tmp_path, "validation"))
    assert [r["model"] for r in rows] == ["deepseek-r1-8b"]
    assert rows[0]["model_checkpoint"] == "deepseek-r1-distill-llama-8b"

    store.merge(tmp_path, "validation", [Source(path=fonte, models=("deepseek-r1-8b",))])
    assert [r["model"] for r in store.stream_outcomes(tmp_path, "validation")] == ["deepseek-r1-8b"]


def test_destilacao_aposentada_nao_entra_no_arquivo_definitivo(tmp_path):
    """O 0528 sobre Qwen3 não é o DeepSeek da grade, nem um décimo aluno.

    Ele rodou no lugar do DeepSeek de 8B num run inteiro. Se as linhas dele
    passassem, ou seriam contadas como o modelo que não são, ou apareceriam como
    um aluno a mais que a grade não pede.
    """
    fonte = escreve(tmp_path / "f.jsonl", [
        linha(model="deepseek-r1-0528-qwen3-8b"),
        linha(model="phi2"),
    ])
    report = store.merge(tmp_path, "validation", [
        Source(path=fonte, drop_models=grid.RETIRED_STUDENTS, stamp={"source_run": "r"})])
    assert report.rejected_by_filter == 1
    assert [r["model"] for r in store.stream_outcomes(tmp_path, "validation")] == ["phi2"]


def test_professor_historico_tambem_e_canonico(tmp_path):
    fonte = escreve(tmp_path / "t.jsonl", [
        linha(condition="teacher_simple", teacher_model="deepseek-r1-distill-llama-8b")])
    store.merge(tmp_path, "validation", [Source(path=fonte)])
    linha_lida = next(iter(store.stream_outcomes(tmp_path, "validation")))
    assert linha_lida["teacher_model"] == "deepseek-r1-8b"
    assert linha_lida["teacher_checkpoint"] == "deepseek-r1-distill-llama-8b"
