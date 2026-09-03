"""
Backend falso e determinístico, sem GPU nem pesos.

Existe para uma coisa: rodar o pipeline inteiro — baseline, reflexão, índice,
avaliação, controles e análise — e verificar que a plumbaria está certa antes de
gastar dezenas de horas de GPU. Testa retomada, chaves, ordem dos resultados,
schema do JSONL e as métricas, tudo em segundos.

O comportamento é uma caricatura útil, não aleatório: a resposta escolhida é uma
função hash do prompt, então o mesmo prompt sempre dá a mesma letra, e a
acurácia fica um pouco acima do chute. Prompts que contêm reflexões recuperadas
acertam com probabilidade maior, para que a etapa de análise tenha um efeito
mensurável para medir e as métricas de utility e transferability sejam
exercitadas de verdade.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

from rmcq.backends.base import Backend, Generation, GenParams
from rmcq.store import get_logger

log = get_logger(__name__)

_OPTION_LINE = re.compile(r"^([A-H])\)\s", re.MULTILINE)


def _hash_unit(text: str, salt: str = "") -> float:
    """Float determinístico em [0, 1) a partir do texto."""
    digest = hashlib.sha256((salt + text).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class StubBackend(Backend):
    """
    Sem VRAM, sem download, saída determinística.

    `skill` é a probabilidade base de acerto sem reflexão; `reflection_boost` é
    quanto sobe quando o prompt traz reflexões recuperadas.
    """

    def __init__(
        self,
        model_key: str,
        skill: float = 0.45,
        reflection_boost: float = 0.12,
    ) -> None:
        super().__init__(model_key)
        # A "habilidade" varia por modelo, de forma estável, para que comparar
        # modelos na análise produza uma ordenação e não empate técnico.
        self.skill = min(0.9, skill + 0.10 * _hash_unit(model_key, "skill"))
        self.reflection_boost = reflection_boost
        log.info("StubBackend(%s) skill=%.3f (sem GPU)", model_key, self.skill)

    def count_tokens(self, text: str) -> int:
        # Aproximação grosseira e barata: ~4 caracteres por token.
        return max(1, len(text) // 4)

    def _labels_in(self, prompt: str) -> list[str]:
        # A transfer prompt contains options for the source and target. Only
        # the final block belongs to the question that must be answered.
        target = prompt.split("Now answer the validation question independently.")[-1]
        found = _OPTION_LINE.findall(target)
        return found or list("ABCD")

    def _gold_hint(self, prompt: str) -> str | None:
        """
        O stub não conhece o gabarito, e é de propósito.

        Ele escolhe uma letra por hash do enunciado. A acurácia resultante fica
        perto do chute, com um deslocamento pequeno quando há reflexão. Isso é
        suficiente para exercitar as métricas sem fingir competência.
        """
        return None

    def generate(
        self,
        prompts: Sequence[str],
        params: GenParams,
        system: str | None = None,
        desc: str = "",
    ) -> list[Generation]:
        results: list[Generation] = []

        for prompt in prompts:
            labels = self._labels_in(prompt)
            has_reflection = "<retrieved_training_question>" in prompt
            is_reflection_task = (
                "Correct answer (private feedback):" in prompt
                or "Student's previous answer:" in prompt
            )

            if is_reflection_task:
                depth = "detailed" if "Analyze your reasoning in detail" in prompt or "Write a detailed reflection" in prompt else "brief"
                n_sent = 8 if depth == "detailed" else 4
                body = " ".join(
                    f"Sentence {i} of a {depth} synthetic reflection about the reasoning used."
                    for i in range(1, n_sent + 1)
                )
                text = f"{body} The lesson is to verify each intermediate step before committing."
                results.append(
                    Generation(
                        text=text,
                        prompt_tokens=self.count_tokens(prompt),
                        completion_tokens=self.count_tokens(text),
                        latency_s=0.001,
                        finish_reason="stop",
                    )
                )
                continue

            samples: list[str] = []
            for s in range(max(1, params.n)):
                salt = f"{self.key}|{s}|{'refl' if has_reflection else 'base'}"
                pick = labels[int(_hash_unit(prompt, salt) * len(labels)) % len(labels)]
                if has_reflection and _hash_unit(prompt, salt + "boost") < self.reflection_boost:
                    # Desloca a escolha, para que a condição com reflexão não
                    # seja idêntica ao baseline e utility saia diferente de zero.
                    pick = labels[(labels.index(pick) + 1) % len(labels)]
                samples.append(
                    "Step 1: consider each option.\n"
                    "Step 2: eliminate the implausible ones.\n"
                    f"FINAL ANSWER: {pick}"
                )

            results.append(
                Generation(
                    text=samples[0],
                    prompt_tokens=self.count_tokens(prompt),
                    completion_tokens=self.count_tokens(samples[0]),
                    latency_s=0.001,
                    finish_reason="stop",
                    samples=samples if params.n > 1 else [],
                )
            )

        return results

    def render(self, tokenizer, prompt: str, system: str | None = None) -> str:
        # Sem tokenizer, sem chat template. O prompt passa direto.
        return f"{system}\n\n{prompt}" if system else prompt
