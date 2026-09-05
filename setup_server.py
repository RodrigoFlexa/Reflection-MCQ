#!/usr/bin/env python
"""Install the additions in the active Python environment on GPU or Petrobras."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def pip_install(requirements, preserve_gpu=False):
    command = [sys.executable, "-m", "pip", "install", "-r", str(ROOT / requirements)]
    # Keep the working GPU stack. Do not upgrade torch/vLLM to add one dataset.
    with tempfile.TemporaryDirectory() as folder:
        if preserve_gpu:
            pins = []
            for package in ("torch", "vllm", "transformers", "tokenizers", "huggingface-hub", "numpy", "safetensors"):
                try:
                    pins.append(f"{package}=={importlib.metadata.version(package)}")
                except importlib.metadata.PackageNotFoundError:
                    pass
            path = Path(folder) / "constraints.txt"
            path.write_text("\n".join(pins) + "\n", encoding="utf-8")
            command += ["-c", str(path)]
        subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", choices=("gpu", "petrobras"))
    parser.add_argument("--experiment-id")
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--install-gpu-stack", action="store_true", help="Install full requirements only in a new GPU environment.")
    args = parser.parse_args()
    if args.server == "gpu":
        if args.install_gpu_stack:
            pip_install("requirements.txt")
        else:
            pip_install("requirements-data.txt", preserve_gpu=True)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        import rmcq  # load .env before importing any CUDA libraries
        import torch
        import transformers
        import vllm
        from packaging.version import Version
        if Version(vllm.__version__) < Version("0.8.5") or Version(transformers.__version__) < Version("4.51.0"):
            raise RuntimeError("The new models require vLLM >= 0.8.5 and Transformers >= 4.51. Update a separate GPU environment.")
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA unavailable on selected GPU {args.gpu}; no experiment was started")
        print(f"GPU {args.gpu}: {torch.cuda.get_device_name(0)}; vLLM {vllm.__version__}; Transformers {transformers.__version__}", flush=True)
        subprocess.run([sys.executable, "prepare_datasets.py", "--datasets", "race"], cwd=ROOT, check=True)
    else:
        # All RACE splits arrive byte-for-byte from the GPU; no Hub/CUDA dependency.
        command = [sys.executable, "experiment_ops.py", "restore", "prepare"]
        if args.experiment_id:
            command += ["--experiment-id", args.experiment_id]
        subprocess.run(command, cwd=ROOT, check=True)
        pip_install("requirements-azure.txt")
        import rmcq
        from rmcq.config import AZURE_API_KEY_VAR, AZURE_BASE_URL_VAR, AZURE_ENDPOINT_VAR
        if not os.environ.get(AZURE_API_KEY_VAR) or not (os.environ.get(AZURE_BASE_URL_VAR) or os.environ.get(AZURE_ENDPOINT_VAR)):
            raise RuntimeError("RACE is installed. Configure the Azure key and endpoint in .env before starting teacher.")
        print("Petrobras: RACE installed from GPU handoff; Azure dependencies and credential presence checked.")
    print("Setup complete. No model generation was started.")


if __name__ == "__main__":
    main()
