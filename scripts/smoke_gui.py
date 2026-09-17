"""Exercise real application initialization and direct image inference offscreen."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import tempfile
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication
from src.gui import BirdEcologyWindow


def main():
    app = QApplication.instance() or QApplication([])
    window = BirdEcologyWindow()
    window.show()
    app.processEvents()
    window.dataset_table.selectRow(0)
    app.processEvents()
    with tempfile.TemporaryDirectory() as temporary:
        photo = Path(temporary) / "synthetic-bird.jpg"
        pixels = np.full((150, 200, 3), 215, dtype=np.uint8)
        cv2.ellipse(pixels, (100, 75), (35, 55), 15, 0, 360, (35, 85, 175), -1)
        cv2.imencode(".jpg", pixels)[1].tofile(photo)
        result, *_ = window.analyzer.analyze_path(photo)
        if len(result.regressed_traits) != 7 or not np.isfinite(list(result.regressed_traits.values())).all():
            raise ValueError("直接图像回归未产生七个有限值")
        window.current_path = photo
        window.analysis_path.setText(str(photo))
        window.tabs.setCurrentIndex(1)
        window.run_analysis()
        app.processEvents()
        output = ROOT / "build"
        output.mkdir(exist_ok=True)
        if not window.grab().save(str(output / "gui-smoke.png")):
            raise RuntimeError("界面截图写入失败")
    evidence = {"passed": True, "tabs": window.tabs.count(), "metadata_rows": len(window.metadata),
                "synthetic_image_regression_outputs": len(result.regressed_traits),
                "scope": "界面初始化及推理流程检查；合成图不用于准确率评价"}
    (output / "gui-smoke.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    window.close()
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
