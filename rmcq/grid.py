"""A grade final do experimento, num lugar só.

Quem é aluno, quem é professor e quem ensina quem. Todo o resto do código
pergunta aqui em vez de repetir listas: `run_experiment.py` para decidir o que
gerar, `results_store.py` para dizer o que ainda falta, os notebooks para saber
quais células deveriam existir.

Regra da etapa final (decidida em 12/09/2026):

  - Os quatro professores abertos são modelos da faixa 7B-8B e ensinam apenas
    os alunos de até 3B. Não existe 8B ensinando 8B: um professor do mesmo
    porte do aluno não é professor, é par, e a comparação perderia sentido.
  - O GPT-5.4 é externo à faixa de tamanho, então ensina a grade inteira e
    continua sendo a referência contra a qual os professores abertos são lidos.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Alunos
# ---------------------------------------------------------------------------

# As nove chaves são as de `rmcq.config.MODELS`. A tag do Ollama que aparece no
# protocolo está ao lado só para conferência humana; o que roda é a chave.
STUDENTS: tuple[str, ...] = (
    "phi2",              # phi2:mini
    "deepseek-r1-8b",    # deepseek-r1:8b
    "deepseek-r1-1.5b",  # deepseek-r1:1.5b
    "llama3.1-8b",       # llama3.1:8b
    "llama3.2-3b",       # llama3.2:3b
    "qwen2.5-3b",        # qwen2.5:3b
    "qwen2.5-7b",        # qwen2.5:7b
    "ministral-3-3b",    # ministral-3:3b
    "ministral-3-8b",    # ministral-3:8b
)

# Alunos de até 3B: os que recebem reflexão dos professores abertos.
SMALL_STUDENTS: tuple[str, ...] = (
    "phi2",
    "deepseek-r1-1.5b",
    "llama3.2-3b",
    "qwen2.5-3b",
    "ministral-3-3b",
)

# ---------------------------------------------------------------------------
# Identidade dos modelos
# ---------------------------------------------------------------------------

# O DeepSeek-R1 entra na grade pela destilação sobre Llama, por padronização.
# Onde há escolha, é essa a base; os nomes canônicos escondem a destilação
# porque ela é constante, não porque seja indiferente.
#
# O 1.5B é a exceção forçada: a família Distill-Llama oficial tem 8B e 70B, e
# nada entre os dois — `DeepSeek-R1-Distill-Llama-1.5B` não existe. Naquele
# tamanho, Qwen é a única destilação disponível.
#
# `DeepSeek-R1-0528-Qwen3-8B` ficou DE FORA: é outra base, e o experimento
# padronizou a de Llama. Ele continua registrado em `rmcq.config.MODELS` para
# que o run que o usou continue legível, mas não é mais o DeepSeek da grade, e
# a consolidação não conta as linhas dele.
#
# O mapa vale para RÓTULO de resultado. Para carregar pesos, cada chave
# histórica continua apontando para o seu próprio checkpoint: um run antigo
# afirma ter rodado aquele checkpoint, e isso não muda retroativamente. A linha
# consolidada guarda o checkpoint de origem em `model_checkpoint`.
MODEL_ALIASES: dict[str, str] = {
    "deepseek-r1-distill-llama-8b": "deepseek-r1-8b",
    "deepseek-r1-distill-qwen-1.5b": "deepseek-r1-1.5b",
}

# Modelos que já foram alunos e não são mais. A consolidação recusa as linhas
# deles: sem esta lista elas entrariam no arquivo definitivo como um décimo
# modelo, ou pior, seriam confundidas com o DeepSeek da grade.
RETIRED_STUDENTS: tuple[str, ...] = (
    "deepseek-r1-0528-qwen3-8b",
)


def canonical(model: str | None) -> str | None:
    """Nome pelo qual este modelo é contado na grade."""
    if model is None:
        return None
    return MODEL_ALIASES.get(model, model)


def historical_names(model: str) -> tuple[str, ...]:
    """Todos os nomes sob os quais este modelo já apareceu, o canônico incluso."""
    return (model, *sorted(k for k, v in MODEL_ALIASES.items() if v == model))

# ---------------------------------------------------------------------------
# Professores
# ---------------------------------------------------------------------------

EXTERNAL_TEACHER = "gpt-5-4-petrobras"

OPEN_TEACHERS: tuple[str, ...] = (
    "deepseek-r1-8b",
    "llama3.1-8b",
    "qwen2.5-7b",
    "ministral-3-8b",
)

TEACHERS: tuple[str, ...] = (EXTERNAL_TEACHER, *OPEN_TEACHERS)

# O professor externo fala por API; os abertos carregam pesos na GPU como
# qualquer aluno. Quem chama `get_backend` decide o kind a partir disso, mas
# sempre deixando uma opção de sobreposição — os testes apontam o externo para
# o stub, e um backend fixo aqui tornaria isso impossível.
TEACHER_BACKEND_KIND: dict[str, str] = {EXTERNAL_TEACHER: "azure"}

# ---------------------------------------------------------------------------
# Condições
# ---------------------------------------------------------------------------

SELF_CONDITIONS: tuple[str, ...] = ("baseline", "self_simple", "self_complex")
TEACHER_CONDITIONS: tuple[str, ...] = ("teacher_simple", "teacher_complex")
ALL_CONDITIONS: tuple[str, ...] = SELF_CONDITIONS + TEACHER_CONDITIONS

# Linhas antigas, de quando existia um professor só, não trazem `teacher_model`.
# Naquele desenho o professor era sempre o GPT-5.4, então a ausência do campo é
# uma afirmação, não um dado perdido.
LEGACY_TEACHER = EXTERNAL_TEACHER


def is_teacher_condition(condition: str) -> bool:
    return condition in TEACHER_CONDITIONS


def students_for(teacher: str) -> tuple[str, ...]:
    """Alunos que este professor ensina."""
    if teacher == EXTERNAL_TEACHER:
        return STUDENTS
    if teacher in OPEN_TEACHERS:
        return SMALL_STUDENTS
    raise KeyError(f"{teacher!r} não é professor da grade final")


def teachers_for(student: str) -> tuple[str, ...]:
    """Professores que ensinam este aluno."""
    if student not in STUDENTS:
        raise KeyError(f"{student!r} não é aluno da grade final")
    return tuple(t for t in TEACHERS if student in students_for(t))


def teacher_pairs() -> tuple[tuple[str, str], ...]:
    """Todos os pares (professor, aluno) da grade, em ordem estável."""
    return tuple((t, s) for t in TEACHERS for s in students_for(t))


def conditions_for(student: str) -> tuple[tuple[str, str | None], ...]:
    """Células (condição, professor) que este aluno deve ter ao final.

    O professor é `None` nas condições que não têm professor — baseline e as
    duas autorreflexões — para que a célula seja uma chave única e comparável
    com o que está gravado no disco.
    """
    cells: list[tuple[str, str | None]] = [(c, None) for c in SELF_CONDITIONS]
    for teacher in teachers_for(student):
        cells.extend((c, teacher) for c in TEACHER_CONDITIONS)
    return tuple(cells)


def expected_cells() -> tuple[tuple[str, str, str | None], ...]:
    """Grade inteira como (aluno, condição, professor)."""
    return tuple((s, c, t) for s in STUDENTS for c, t in conditions_for(s))


def describe() -> str:
    """Resumo legível da grade, para log e para o cabeçalho dos relatórios."""
    lines = [
        f"Alunos ({len(STUDENTS)}): {', '.join(STUDENTS)}",
        f"Alunos pequenos, ate 3B ({len(SMALL_STUDENTS)}): {', '.join(SMALL_STUDENTS)}",
        f"Professor externo: {EXTERNAL_TEACHER} -> todos os {len(STUDENTS)} alunos",
        f"Professores abertos ({len(OPEN_TEACHERS)}): {', '.join(OPEN_TEACHERS)} -> os {len(SMALL_STUDENTS)} alunos de ate 3B",
        f"Pares professor-aluno: {len(teacher_pairs())}",
        f"Celulas por item de avaliacao: {len(expected_cells())}",
    ]
    return "\n".join(lines)


def validate() -> None:
    """Erros de digitação na grade viram corrupção silenciosa de dados."""
    from rmcq.config import MODELS

    unknown = [m for m in (*STUDENTS, *TEACHERS) if m not in MODELS]
    if unknown:
        raise ValueError(f"Fora de rmcq.config.MODELS: {', '.join(sorted(unknown))}")
    if not set(SMALL_STUDENTS) <= set(STUDENTS):
        raise ValueError("SMALL_STUDENTS precisa ser subconjunto de STUDENTS")
    overlap = set(SMALL_STUDENTS) & set(OPEN_TEACHERS)
    if overlap:
        raise ValueError(f"Um aluno pequeno não pode ser professor aberto: {', '.join(sorted(overlap))}")
    for teacher in OPEN_TEACHERS:
        if teacher in students_for(teacher):
            raise ValueError(f"{teacher} apareceria como professor de si mesmo")
    # Um apelido que aponta para um nome que não é aluno nem professor não
    # renomeia nada: é erro de digitação que só apareceria na hora de contar.
    destinos = set(MODEL_ALIASES.values())
    orfaos = destinos - set(STUDENTS) - set(TEACHERS)
    if orfaos:
        raise ValueError(f"MODEL_ALIASES aponta para modelos fora da grade: {', '.join(sorted(orfaos))}")
    colisao = set(MODEL_ALIASES) & set(STUDENTS)
    if colisao:
        raise ValueError(f"Nome histórico usado como chave da grade: {', '.join(sorted(colisao))}")
    # Um modelo aposentado que ainda tenha apelido voltaria a contar pela porta
    # dos fundos, com o nome de um aluno atual.
    ressuscitados = set(RETIRED_STUDENTS) & (set(MODEL_ALIASES) | set(STUDENTS) | set(TEACHERS))
    if ressuscitados:
        raise ValueError(f"Modelo aposentado ainda em uso: {', '.join(sorted(ressuscitados))}")


if __name__ == "__main__":
    validate()
    print(describe())
