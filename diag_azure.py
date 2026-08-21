#!/usr/bin/env python
"""
Diagnóstico de conexão com o Azure OpenAI.

Para quando a chamada falha e a mensagem do servidor não diz o suficiente —
em especial o `404 Resource Not Found`, que tem três causas bem diferentes e
a mesma mensagem:

  1. O nome do DEPLOYMENT não existe nesse recurso (causa mais comum).
  2. O ENDPOINT aponta para outro recurso, ou veio com caminho sobrando
     (deve ser https://<recurso>.openai.azure.com/ e nada além disso).
  3. A API-VERSION é antiga demais para esse modelo. `2024-10-21` é de outubro
     de 2024 e não conhece a família gpt-5: o deployment existe, mas a rota
     naquela versão não, e o servidor responde 404.

Este script separa as três: lista os deployments que o recurso realmente tem e
depois tenta uma chamada mínima em várias api-versions, dizendo qual funciona.

    python diag_azure.py                     # usa o .env
    python diag_azure.py --deployment X      # testa outro nome
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

import rmcq  # noqa: F401 — carrega o .env
from rmcq.config import (
    AZURE_API_KEY_VAR,
    AZURE_API_VERSION,
    AZURE_ENDPOINT_VAR,
    MODELS,
    azure_deployment,
)

import os

# Da mais nova para a mais antiga. As primeiras são as que suportam a família
# gpt-5 e o parâmetro reasoning_effort.
API_VERSIONS = (
    "2025-04-01-preview",
    "2025-03-01-preview",
    "2025-01-01-preview",
    "2024-12-01-preview",
    "2024-10-21",
    "2024-08-01-preview",
    "2024-06-01",
)

LINHA = "=" * 78


def mascarar(chave: str) -> str:
    if not chave:
        return "(vazia)"
    return f"{chave[:4]}…{chave[-4:]} ({len(chave)} caracteres)"


def http(url: str, chave: str, corpo: dict | None = None, timeout: int = 30):
    """Devolve (status, json_ou_texto). Não levanta em erro HTTP."""
    dados = json.dumps(corpo).encode() if corpo is not None else None
    req = urllib.request.Request(
        url,
        data=dados,
        headers={"api-key": chave, "Content-Type": "application/json"},
        method="POST" if corpo is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        texto = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(texto)
        except json.JSONDecodeError:
            return e.code, texto
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def erro_curto(payload) -> str:
    if isinstance(payload, dict):
        err = payload.get("error", payload)
        if isinstance(err, dict):
            return str(err.get("message") or err.get("code") or err)[:160]
    return str(payload)[:160]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Diagnostica a conexão com o Azure OpenAI.")
    p.add_argument("--deployment", default=None, help="nome do deployment (padrão: o do .env)")
    p.add_argument("--model", default="gpt5", help="chave do modelo em config.MODELS")
    args = p.parse_args(argv)

    endpoint = os.environ.get(AZURE_ENDPOINT_VAR, "").strip().rstrip("/")
    chave = os.environ.get(AZURE_API_KEY_VAR, "").strip()
    alvo = args.deployment or (azure_deployment(args.model) if args.model in MODELS else args.model)

    print(f"\n{LINHA}\n  DIAGNÓSTICO AZURE OPENAI\n{LINHA}")
    print(f"  endpoint     {endpoint or '(VAZIO)'}")
    print(f"  api-key      {mascarar(chave)}")
    print(f"  api-version  {AZURE_API_VERSION}   (do .env)")
    print(f"  deployment   {alvo!r}")
    print(LINHA)

    if not endpoint or not chave:
        print("\n  ERRO: endpoint ou chave ausente no .env. Preencha antes de diagnosticar.\n")
        return 2

    # --- checagem 1: formato do endpoint --------------------------------
    print("\n[1] Formato do endpoint")
    if "/openai" in endpoint or endpoint.count("/") > 2:
        print(f"  ✗ PROBLEMA: o endpoint tem caminho sobrando.")
        print(f"    tem  : {endpoint}")
        print(f"    deve : https://<recurso>.openai.azure.com")
        print("    O SDK acrescenta /openai/deployments/... sozinho.")
    elif not endpoint.startswith("https://"):
        print(f"  ✗ PROBLEMA: deve começar com https://")
    else:
        print(f"  ✓ formato ok")

    # --- checagem 2: quais deployments existem --------------------------
    print("\n[2] Deployments que este recurso realmente tem")
    achou_lista = False
    for ver in ("2023-05-15", "2024-10-21", API_VERSIONS[0]):
        status, payload = http(f"{endpoint}/openai/deployments?api-version={ver}", chave)
        if status == 200 and isinstance(payload, dict):
            nomes = [d.get("id") for d in payload.get("data", [])]
            achou_lista = True
            if nomes:
                for n in nomes:
                    marca = "  <<< o que você configurou" if n == alvo else ""
                    print(f"    - {n}{marca}")
                if alvo not in nomes:
                    print(f"\n  ✗ CAUSA ENCONTRADA: {alvo!r} não está na lista acima.")
                    print("    Corrija RMCQ_AZURE_DEPLOYMENT_GPT5 no .env com um dos nomes listados.")
            else:
                print("    (o recurso não devolveu nenhum deployment)")
            break
    if not achou_lista:
        print("    não consegui listar (a chave de dados costuma não ter essa permissão).")
        print("    Confira o nome no portal do Azure > seu recurso > Deployments.")

    # --- checagem 3: qual api-version funciona --------------------------
    print(f"\n[3] Chamada mínima a {alvo!r}, por api-version")
    corpo = {"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 4000}
    funcionou = []
    for ver in API_VERSIONS:
        url = f"{endpoint}/openai/deployments/{alvo}/chat/completions?api-version={ver}"
        status, payload = http(url, chave, corpo)
        if status == 200:
            print(f"  ✓ {ver:24} FUNCIONA")
            funcionou.append(ver)
        elif status == 400 and "max_completion_tokens" in str(payload):
            # Modelo de chat nesta versão: aceita a rota, recusa o parâmetro.
            print(f"  ~ {ver:24} rota ok, mas é modelo de chat (use max_tokens)")
            funcionou.append(ver)
        else:
            print(f"  ✗ {ver:24} {status}: {erro_curto(payload)}")

    # --- veredito -------------------------------------------------------
    print(f"\n{LINHA}\n  VEREDITO\n{LINHA}")
    if funcionou:
        melhor = funcionou[0]
        print(f"\n  Funciona com api-version {melhor}.")
        if melhor != AZURE_API_VERSION:
            print(f"\n  >>> CORREÇÃO: troque no .env da máquina Petrobras:")
            print(f"        AZURE_OPENAI_API_VERSION={melhor}")
            print(f"      (está como {AZURE_API_VERSION}, que não serve para este deployment)")
        else:
            print("  A api-version do .env já é essa — o problema estava em outro lugar.")
        print()
        return 0

    print("\n  Nenhuma api-version funcionou. Pela ordem de probabilidade:")
    print("    1. O nome do deployment está errado — confira a seção [2] acima")
    print("       e o portal do Azure > seu recurso > Deployments.")
    print("    2. O endpoint aponta para outro recurso (o deployment pode existir,")
    print("       mas em outro).")
    print("    3. A chave é de um recurso diferente do endpoint.")
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
