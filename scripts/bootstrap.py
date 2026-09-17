from __future__ import annotations

import subprocess
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
PYTHON = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env={**os.environ, "PYTHONUTF8": "1"})


def main() -> None:
    if sys.version_info[:2] != (3, 13):
        raise SystemExit("请使用 Python 3.13（验证版本 3.13.9）执行本脚本，不能使用 3.14。")
    if not PYTHON.is_file():
        run([sys.executable, "-m", "venv", str(VENV)])
    run([str(PYTHON), "-m", "pip", "install", "-r", str(ROOT / "requirements.lock")])
    run([str(PYTHON), "-m", "pip", "check"])
    run([str(PYTHON), "-c", "import cv2, PySide6, sklearn; print('OpenCV', cv2.__version__); print('PySide6', PySide6.__version__); print('scikit-learn', sklearn.__version__)"])
    print(f"环境准备完成: {PYTHON}")


if __name__ == "__main__":
    main()
