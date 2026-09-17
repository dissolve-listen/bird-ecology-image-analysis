"""Rebuild the complete models/evaluations and fail if numerical references differ."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(arguments: list[str], log: Path) -> None:
    command = [sys.executable, *arguments]
    print("+", " ".join(command), flush=True)
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text(encoding="utf-8")[-8000:])
        raise RuntimeError(f"步骤失败，完整日志: {log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cached", "full"], default="cached")
    parser.add_argument("--archive")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == ROOT or ROOT.is_relative_to(output):
        parser.error("输出目录不能是项目根目录或其父目录。")
    if args.workers < 1:
        parser.error("workers 必须为正整数")
    output.mkdir(parents=True, exist_ok=True)
    # Never leave a previous success file after a failed rerun.
    summary_path = output / "verification.json"
    summary_path.unlink(missing_ok=True)
    started = time.perf_counter()
    run([str(ROOT / "scripts/verify_bundle.py"), "--inputs-only"], output / "inputs.log")
    run([str(ROOT / "tests/test_core.py")], output / "core-tests.log")
    run(["-m", "unittest", "discover", "-s", "tests", "-v"], output / "unit-tests.log")
    if args.mode == "full":
        preparation = [str(ROOT / "scripts/prepare_data.py")]
        if args.archive:
            preparation += ["--archive", args.archive]
        if args.download:
            preparation += ["--download"]
        run(preparation, output / "data-preparation.log")
    training = [str(ROOT / "scripts/train_and_evaluate.py"), "--output-dir", str(output),
                "--workers", str(args.workers),
                "--force-features" if args.mode == "full" else "--cached-features"]
    run(training, output / "training.log")
    from check_results import check_results
    summary = check_results(output)
    # Cached runs must not claim that an older full-run output was recomputed.
    summary["all_raw_image_features_recomputed"] = args.mode == "full"
    summary.update({"mode": args.mode, "python": sys.version, "platform": platform.platform(),
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                    "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
