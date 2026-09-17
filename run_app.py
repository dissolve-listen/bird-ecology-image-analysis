from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from src.config import PATHS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动鸟类生态图像评价系统。")
    parser.add_argument(
        "--demo-screenshot",
        type=Path,
        help="运行一次真实单图分析并保存界面截图（用于自动化验收）。",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    app = QApplication(sys.argv)
    if not (PATHS.metadata_csv.is_file() and PATHS.model_path.is_file()):
        QMessageBox.critical(None, "项目尚未构建", "未找到数据索引或模型。请先按 README 运行 prepare_data.py 和 train_and_evaluate.py。")
        return 2
    from src.gui import BirdEcologyWindow
    window = BirdEcologyWindow()
    window.show()

    if args.demo_screenshot:
        # 自动化验收入口：使用项目内真实的留出样例，不替代人工界面流程。
        def capture_demo() -> None:
            demo_dir = PATHS.project_root / "results" / "demo_batch_images"
            candidates = sorted(demo_dir.glob("*.jpg"))
            if not candidates:
                candidates = sorted((PATHS.cub_root / "images").rglob("*.jpg"))[:1]
            if candidates:
                window.current_path = candidates[0]
                window.analysis_path.setText(str(candidates[0]))
                window.tabs.setCurrentIndex(1)
                window.run_analysis()

            def save_and_exit() -> None:
                args.demo_screenshot.parent.mkdir(parents=True, exist_ok=True)
                window.grab().save(str(args.demo_screenshot))
                app.quit()

            QTimer.singleShot(1000, save_and_exit)

        QTimer.singleShot(200, capture_demo)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
