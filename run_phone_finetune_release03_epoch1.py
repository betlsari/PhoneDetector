from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from validate_phone_finetune_dataset import validate_dataset


ROOT = Path(__file__).resolve().parent
EXPECTED_NAMES = ["0", "cell phone"]
EXPECTED_ORIGINAL_SHA256 = (
    "12dd283d7ae86f953cba2d8741733abc11b9e27c3aeb5c128d6bee8536f345bb"
)

TRAINING_CODE = r"""
import json
import sys
import time
from pathlib import Path
from ultralytics import YOLO

weights, data_yaml, project, run_name = sys.argv[1:5]
model = YOLO(weights, task="detect")
print("MODEL_LOADED", json.dumps({"names": model.names}), flush=True)

batch_state = {"start": None, "index": 0, "total": 0}
def on_train_start(trainer):
    batch_state["index"] = 0
    batch_state["total"] = len(trainer.train_loader)

def on_batch_start(trainer):
    batch_state["start"] = time.perf_counter()

def on_batch_end(trainer):
    if batch_state["start"] is None:
        return
    elapsed = time.perf_counter() - batch_state["start"]
    batch_state["index"] += 1
    index = batch_state["index"]
    total = batch_state["total"]
    if index == 1 or index % 10 == 0 or (total and index == total):
        print(
            "BATCH_PROGRESS",
            json.dumps({
                "epoch": int(getattr(trainer, "epoch", 0)) + 1,
                "batch": index,
                "batches": total,
                "batch_seconds": round(elapsed, 4),
            }),
            flush=True,
        )

model.add_callback("on_train_start", on_train_start)
model.add_callback("on_train_batch_start", on_batch_start)
model.add_callback("on_train_batch_end", on_batch_end)
result = model.train(
    data=data_yaml,
    epochs=1,
    device="cpu",
    imgsz=320,
    batch=2,
    workers=0,
    cache=False,
    amp=False,
    project=project,
    name=run_name,
    exist_ok=False,
    plots=False,
    patience=1,
    seed=42,
    deterministic=True,
    fraction=1.0,
    val=True,
)
save_dir = Path(result.save_dir)
best = save_dir / "weights" / "best.pt"
last = save_dir / "weights" / "last.pt"
if not best.is_file() or not last.is_file():
    raise FileNotFoundError(f"Training ended without both checkpoints: {best}, {last}")
print("TRAINING_COMPLETED", json.dumps({
    "save_dir": str(save_dir),
    "best_pt": str(best),
    "last_pt": str(last),
}), flush=True)

print("CLASS1_VALIDATION_START", flush=True)
trained = YOLO(str(best), task="detect")
metrics = trained.val(
    data=data_yaml,
    split="val",
    imgsz=320,
    batch=2,
    workers=0,
    device="cpu",
    plots=False,
    classes=[1],
    project=project,
    name=run_name + "_class1_val",
    exist_ok=False,
)
box = metrics.box
class_metrics = {}
for position, class_id in enumerate(list(getattr(box, "ap_class_index", []))):
    if int(class_id) == 1:
        class_metrics = {
            "precision": float(box.p[position]),
            "recall": float(box.r[position]),
            "mAP50": float(box.ap50[position]),
            "mAP50_95": float(box.ap[position]),
        }
if not class_metrics:
    raise RuntimeError("Validation did not return metrics for class id 1.")
summary = {
    "best_pt": str(best),
    "last_pt": str(last),
    "cell_phone_class_id": 1,
    "cell_phone_metrics": class_metrics,
    "validation_split": "val",
}
(save_dir / "cell_phone_validation_metrics.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print("CLASS1_VALIDATION_METRICS", json.dumps(summary), flush=True)
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a logged, one-epoch CPU fine-tuning attempt with a hard timeout."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, choices=[1], required=True)
    parser.add_argument("--device", choices=["cpu"], required=True)
    parser.add_argument("--imgsz", type=int, choices=[320], required=True)
    parser.add_argument("--batch", type=int, choices=[2], required=True)
    parser.add_argument("--workers", type=int, choices=[0], required=True)
    parser.add_argument("--cache", choices=["false"], required=True)
    parser.add_argument("--amp", choices=["false"], required=True)
    parser.add_argument("--timeout-seconds", type=int, choices=[1800], required=True)
    parser.add_argument("--log", type=Path, required=True)
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    model = args.model.resolve()
    data_yaml = args.data.resolve()
    project = args.project.resolve()
    if not model.is_file():
        raise FileNotFoundError(f"Starting weights are missing: {model}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset config is missing: {data_yaml}")
    if sha256(model) != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError("Original best.pt does not match the previously verified hash.")
    config: Any = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Dataset YAML must contain a mapping.")
    if config.get("nc") != 2 or config.get("names") != EXPECTED_NAMES:
        raise ValueError(
            f"Unexpected class mapping: nc={config.get('nc')}, names={config.get('names')}"
        )
    dataset_root = Path(str(config.get("path", data_yaml.parent))).resolve()
    if dataset_root != (ROOT / "custom_dataset" / "phone_finetune_release_03_corrected").resolve():
        raise ValueError(f"Refusing unexpected dataset root: {dataset_root}")
    validation = validate_dataset(dataset_root)
    if validation.get("status") != "PASS":
        raise RuntimeError("Dataset validator did not pass.")
    save_dir = project / args.name
    if save_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing training run: {save_dir}")
    log_path = args.log.resolve()
    if log_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing log: {log_path}")
    return model, data_yaml


def clean_progress(value: str) -> str:
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    return " ".join(value.replace("\r", " ").replace("\n", " ").split())[-500:]


def run() -> int:
    args = parse_args()
    model, data_yaml = validate_inputs(args)
    project = args.project.resolve()
    log_path = args.log.resolve()
    project.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    model_hash = sha256(model)
    metadata = {
        "model": str(model),
        "model_sha256_before": model_hash,
        "data": str(data_yaml),
        "dataset_root": str(data_yaml.parent),
        "output": str(project / args.name),
        "epochs": args.epochs,
        "device": args.device,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "cache": False,
        "amp": False,
        "timeout_seconds": args.timeout_seconds,
        "training_started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("Starting controlled training:", json.dumps(metadata, ensure_ascii=False))

    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    child = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            TRAINING_CODE,
            str(model),
            str(data_yaml),
            str(project),
            args.name,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        creationflags=creationflags,
    )
    chunks: queue.Queue[bytes | None] = queue.Queue()

    def read_child_output() -> None:
        assert child.stdout is not None
        try:
            while True:
                chunk = os.read(child.stdout.fileno(), 65536)
                if not chunk:
                    break
                chunks.put(chunk)
        finally:
            chunks.put(None)

    reader = threading.Thread(target=read_child_output, daemon=True)
    reader.start()
    started = time.monotonic()
    deadline = started + args.timeout_seconds
    last_heartbeat = started
    last_progress = "waiting for first training output"
    timed_out = False
    reader_finished = False
    log_path.write_text(
        "CONTROLLED TRAIN LOG\n"
        + json.dumps(metadata, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    with log_path.open("a", encoding="utf-8", errors="replace", newline="") as log:
        while child.poll() is None:
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                timeout_line = (
                    f"\nTIMEOUT after {args.timeout_seconds}s; sending CTRL_BREAK "
                    f"to training PID {child.pid}.\n"
                )
                log.write(timeout_line)
                log.flush()
                print(timeout_line.strip(), flush=True)
                if os.name == "nt":
                    try:
                        child.send_signal(signal.CTRL_BREAK_EVENT)
                    except OSError as exc:
                        log.write(f"CTRL_BREAK failed: {exc!r}\n")
                        log.flush()
                else:
                    child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    log.write(
                        f"Graceful stop timed out; terminating training PID {child.pid}.\n"
                    )
                    log.flush()
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        log.write(
                            f"Terminate timed out; killing only training PID {child.pid}.\n"
                        )
                        log.flush()
                        child.kill()
                        child.wait()
                break

            try:
                item = chunks.get(timeout=min(1.0, max(0.05, deadline - now)))
            except queue.Empty:
                item = b""
            if item is None:
                reader_finished = True
            elif item:
                text = item.decode("utf-8", errors="replace")
                log.write(text)
                log.flush()
                fragments = re.split(r"[\r\n]+", clean_progress(text))
                for fragment in fragments:
                    if fragment:
                        last_progress = fragment
                sys.stdout.write(text)
                sys.stdout.flush()

            now = time.monotonic()
            if now - last_heartbeat >= 30 and child.poll() is None:
                elapsed = now - started
                heartbeat = (
                    f"\nWATCHDOG elapsed={elapsed:.0f}s pid={child.pid} "
                    f"last_progress={last_progress}\n"
                )
                log.write(heartbeat)
                log.flush()
                print(heartbeat.strip(), flush=True)
                last_heartbeat = now
            if reader_finished and child.poll() is not None:
                break

        while True:
            try:
                item = chunks.get_nowait()
            except queue.Empty:
                break
            if item:
                text = item.decode("utf-8", errors="replace")
                log.write(text)
                log.flush()
                sys.stdout.write(text)
                sys.stdout.flush()

        exit_code = child.wait()
        model_hash_after = sha256(model)
        log.write(
            f"\nPROCESS_EXIT code={exit_code} timed_out={timed_out} "
            f"original_model_unchanged={model_hash_after == model_hash}\n"
        )
        log.flush()

    print(
        f"Training process ended: pid={child.pid}, exit_code={exit_code}, "
        f"timed_out={timed_out}, log={log_path}"
    )
    print(f"Original best.pt unchanged: {model_hash_after == model_hash}")
    return 124 if timed_out else exit_code


if __name__ == "__main__":
    raise SystemExit(run())
