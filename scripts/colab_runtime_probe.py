from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from common import write_json
from wandb_support import wandb_run_context


JsonDict = Dict[str, Any]


def detect_runtime() -> JsonDict:
    runtime: JsonDict = {
        "is_colab": "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ,
        "has_tpu_env": any(key.startswith("TPU_") for key in os.environ),
        "colab_gpu_env": os.environ.get("COLAB_GPU"),
    }

    try:
        import torch
    except Exception:
        runtime["torch_available"] = False
        runtime["cuda_available"] = False
        runtime["cuda_device_count"] = 0
        return runtime

    runtime["torch_available"] = True
    runtime["cuda_available"] = bool(torch.cuda.is_available())
    runtime["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        runtime["cuda_device_name"] = str(torch.cuda.get_device_name(0))
        total_memory = torch.cuda.get_device_properties(0).total_memory
        runtime["cuda_total_memory_gb"] = round(float(total_memory) / float(1024**3), 2)

    nvidia_smi = shutil.which("nvidia-smi")
    runtime["nvidia_smi_available"] = bool(nvidia_smi)
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            )
            runtime["nvidia_smi_summary"] = result.stdout.strip().splitlines()
        except Exception as exc:
            runtime["nvidia_smi_error"] = str(exc)

    if runtime.get("cuda_available"):
        runtime["recommended_lane"] = "gpu"
    elif runtime.get("has_tpu_env"):
        runtime["recommended_lane"] = "tpu_experimental"
    else:
        runtime["recommended_lane"] = "cpu_or_setup_required"
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Colab runtime capabilities and emit a JSON summary")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    output_dir = Path(args.output_json).resolve().parent if args.output_json else None

    with wandb_run_context(
        config=None,
        output_dir=output_dir,
        script_name="colab_runtime_probe.py",
        job_type="runtime_probe",
        extra_config={"output_json": str(Path(args.output_json).resolve()) if args.output_json else None},
    ) as wandb_session:
        runtime = detect_runtime()
        if args.output_json:
            write_json(Path(args.output_json), runtime)
        else:
            import json

            print(json.dumps(runtime, indent=2))

        wandb_session.log_metrics(runtime, prefix="runtime")
        wandb_session.update_summary(runtime, prefix="runtime")
        if args.output_json:
            wandb_session.log_output_artifact(
                output_dir=Path(args.output_json).resolve().parent,
                candidate_files=[Path(args.output_json).name],
                artifact_type="runtime_probe_outputs",
                metadata=runtime,
            )


if __name__ == "__main__":
    main()
