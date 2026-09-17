from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .analysis import BirdAnalyzer
from .config import PATHS
from .dataset import read_portable_metadata
from .processing import overlay_mask


def _pixmap(image: np.ndarray, width: int = 360, height: int = 260) -> QPixmap:
    if image.ndim == 2:
        rendered = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rendered = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, channels = rendered.shape
    qimage = QImage(rendered.data, w, h, channels * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage).scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class BirdEcologyWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        # Offscreen Qt may not discover Windows' Chinese fallback fonts.
        windows_directory = os.environ.get("WINDIR")
        if windows_directory:
            font_file = Path(windows_directory) / "Fonts/msyh.ttc"
            if font_file.is_file():
                QFontDatabase.addApplicationFont(str(font_file))
                QApplication.instance().setFont(QFont("Microsoft YaHei", 9))
        self.setWindowTitle("鸟类生态图像特征工程与量化评价系统")
        self.resize(1380, 860)
        self.metadata = read_portable_metadata(PATHS.metadata_csv, PATHS.cub_root)
        self.analyzer = BirdAnalyzer()
        self.current_path: Path | None = None
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self._build_browser_tab()
        self._build_analysis_tab()
        self._build_batch_tab()
        self._build_stats_tab()
        self._build_help_tab()
        self.search_dataset()

    def _build_browser_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("按鸟种或文件名检索"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("例如 Albatross、Black 或图片编号")
        self.search_edit.returnPressed.connect(self.search_dataset)
        controls.addWidget(self.search_edit, 1)
        search_button = QPushButton("检索")
        search_button.clicked.connect(self.search_dataset)
        controls.addWidget(search_button)
        self.browser_status = QLabel()
        controls.addWidget(self.browser_status)
        layout.addLayout(controls)
        splitter = QSplitter()
        self.dataset_table = QTableWidget(0, 5)
        self.dataset_table.setHorizontalHeaderLabels(["ID", "鸟种", "文件", "划分", "类别"])
        self.dataset_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.dataset_table.itemSelectionChanged.connect(self.preview_dataset_item)
        splitter.addWidget(self.dataset_table)
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        self.browser_preview = QLabel("选择一行以预览数据集图像")
        self.browser_preview.setAlignment(Qt.AlignCenter)
        self.browser_preview.setMinimumSize(420, 350)
        preview_layout.addWidget(self.browser_preview)
        self.browser_details = QPlainTextEdit()
        self.browser_details.setReadOnly(True)
        preview_layout.addWidget(self.browser_details)
        open_selected = QPushButton("在单图分析页打开所选图像")
        open_selected.clicked.connect(self.open_selected_dataset_image)
        preview_layout.addWidget(open_selected)
        splitter.addWidget(preview_panel)
        splitter.setSizes([800, 480])
        layout.addWidget(splitter)
        self.tabs.addTab(page, "数据集浏览与检索")

    def search_dataset(self) -> None:
        query = self.search_edit.text().strip().lower() if hasattr(self, "search_edit") else ""
        candidates = self.metadata
        if query:
            mask = candidates["display_name"].str.lower().str.contains(query, na=False) | candidates["relative_path"].str.lower().str.contains(query, na=False) | candidates["image_id"].astype(str).str.contains(query)
            candidates = candidates.loc[mask]
        candidates = candidates.head(150)
        self.dataset_rows = candidates.reset_index(drop=True)
        self.dataset_table.setRowCount(len(self.dataset_rows))
        for row_index, row in self.dataset_rows.iterrows():
            values = [row["image_id"], row["display_name"], Path(row["relative_path"]).name, "训练集" if row["split"] == "train" else "测试集", row["class_id"]]
            for column, value in enumerate(values):
                self.dataset_table.setItem(row_index, column, QTableWidgetItem(str(value)))
        self.dataset_table.resizeColumnsToContents()
        self.browser_status.setText(f"显示 {len(self.dataset_rows)} 条，完整数据集 {len(self.metadata)} 条")

    def preview_dataset_item(self) -> None:
        selected = self.dataset_table.selectedItems()
        if not selected:
            return
        row = self.dataset_rows.iloc[selected[0].row()]
        if not Path(row["image_path"]).is_file():
            self.browser_preview.setText("原始图像尚未准备。请运行 scripts/prepare_data.py --download。\n也可以在单图分析页打开自己的鸟类照片。")
            return
        image = cv2.imdecode(np.fromfile(row["image_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            self.browser_preview.setPixmap(_pixmap(image, 440, 400))
        self.browser_details.setPlainText(
            f"鸟种: {row['display_name']}\n类别编号: {row['class_id']}\n数据划分: {row['split']}\n"
            f"官方目标框: ({row['bbox_x']:.1f}, {row['bbox_y']:.1f}, {row['bbox_width']:.1f}, {row['bbox_height']:.1f})\n"
            "提示: 目标框只用于离线评估，不输入分割算法。"
        )

    def open_selected_dataset_image(self) -> None:
        selected = self.dataset_table.selectedItems()
        if not selected:
            return
        self.current_path = Path(self.dataset_rows.iloc[selected[0].row()]["image_path"])
        self.analysis_path.setText(str(self.current_path))
        self.tabs.setCurrentIndex(1)
        self.run_analysis()

    def _image_label(self, title: str) -> tuple[QLabel, QWidget]:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(QLabel(title))
        label = QLabel("尚未处理")
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(320, 280)
        label.setStyleSheet("border: 1px solid #c8d0d8; background: #f7f9fb;")
        layout.addWidget(label)
        return label, container

    def _build_analysis_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        self.analysis_path = QLineEdit()
        self.analysis_path.setReadOnly(True)
        self.analysis_path.setPlaceholderText("选择一张鸟类图像")
        controls.addWidget(self.analysis_path, 1)
        choose = QPushButton("选择图像")
        choose.clicked.connect(self.choose_image)
        analyze = QPushButton("执行图像处理与生态评价")
        analyze.clicked.connect(self.run_analysis)
        controls.addWidget(choose)
        controls.addWidget(analyze)
        layout.addLayout(controls)
        image_grid = QGridLayout()
        self.original_label, original_panel = self._image_label("原始图像")
        self.filtered_label, filtered_panel = self._image_label("去噪与滤波")
        self.mask_label, mask_panel = self._image_label("OpenCV 前景分割")
        self.overlay_label, overlay_panel = self._image_label("分割结果叠加")
        for index, panel in enumerate((original_panel, filtered_panel, mask_panel, overlay_panel)):
            image_grid.addWidget(panel, index // 2, index % 2)
        layout.addLayout(image_grid)
        self.analysis_text = QPlainTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMinimumHeight(190)
        layout.addWidget(self.analysis_text)
        self.tabs.addTab(page, "单图生态评价")

    def choose_image(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "选择鸟类图像", str(PATHS.cub_root / "images"), "图像文件 (*.jpg *.jpeg *.png *.bmp)")
        if filename:
            self.current_path = Path(filename)
            self.analysis_path.setText(filename)

    def run_analysis(self) -> None:
        if not self.current_path:
            QMessageBox.information(self, "尚未选择图像", "请先选择一张图像，或在数据集浏览页选择图像。")
            return
        try:
            result, original, mask, filtered = self.analyzer.analyze_path(self.current_path)
            self.original_label.setPixmap(_pixmap(original))
            self.filtered_label.setPixmap(_pixmap(filtered))
            self.mask_label.setPixmap(_pixmap(mask))
            self.overlay_label.setPixmap(_pixmap(overlay_mask(original, mask)))
            top5 = "\n".join(f"  {index + 1}. {item['species']}  {item['probability']:.2%}" for index, item in enumerate(result.top5))
            traits = "\n".join(f"  {key}: {value}" for key, value in result.ecological_traits.items()) or "  AVONET 指标文件不可用"
            warning = f"\n注意: {result.warning}" if result.warning else ""
            regression = "\n".join(f"  {key}: {value:.3f}" for key, value in result.regressed_traits.items()) or "  回归模型尚未训练"
            self.analysis_text.setPlainText(
                f"图像直接回归的形态性状:\n{regression}\n{result.regression_note}\n\n"
                f"预测鸟种: {result.predicted_species}\n分类置信度: {result.confidence:.2%}\n分割质量评分: {result.segmentation_quality:.2%}\n"
                f"预测框: {result.predicted_bbox}\n\nTop-5 预测:\n{top5}\n\n按预测鸟种查表的 AVONET 参考值:\n{traits}{warning}"
            )
        except Exception as error:
            QMessageBox.critical(self, "分析失败", str(error))

    def _build_batch_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        self.batch_dir = QLineEdit()
        self.batch_dir.setReadOnly(True)
        choose = QPushButton("选择图片文件夹")
        choose.clicked.connect(self.choose_batch_folder)
        run = QPushButton("批量处理并导出 CSV")
        run.clicked.connect(self.run_batch)
        controls.addWidget(self.batch_dir, 1)
        controls.addWidget(choose)
        controls.addWidget(run)
        layout.addLayout(controls)
        self.batch_table = QTableWidget(0, 7)
        self.batch_table.setHorizontalHeaderLabels(["文件", "预测鸟种", "分类置信度", "分割质量", "回归体重 g", "回归翅长 mm", "提示"])
        layout.addWidget(self.batch_table)
        self.batch_status = QLabel("支持 JPG、PNG、BMP；结果将保存到 results/batch_results.csv")
        layout.addWidget(self.batch_status)
        self.tabs.addTab(page, "批量处理与导出")

    def choose_batch_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择包含待处理图像的文件夹", str(PATHS.cub_root / "images"))
        if folder:
            self.batch_dir.setText(folder)

    def run_batch(self) -> None:
        if not self.batch_dir.text():
            QMessageBox.information(self, "尚未选择文件夹", "请先选择待处理图片文件夹。")
            return
        paths = [path for path in Path(self.batch_dir.text()).rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        if not paths:
            QMessageBox.information(self, "未找到图像", "所选文件夹中没有支持的图像文件。")
            return
        paths = paths[:300]
        csv_rows = []
        preview_rows = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for path in paths:
                result, *_ = self.analyzer.analyze_path(path)
                csv_rows.append(result.to_csv_row())
                preview_rows.append({"文件": path.name, "预测鸟种": result.predicted_species,
                    "置信度": f"{result.confidence:.2%}", "分割质量": f"{result.segmentation_quality:.2%}",
                    "回归体重": f"{result.regressed_traits['体重_g']:.3f}" if result.regressed_traits else "未训练",
                    "回归翅长": f"{result.regressed_traits['翅长_mm']:.3f}" if result.regressed_traits else "未训练",
                    "提示": result.warning or ""})
        finally:
            QApplication.restoreOverrideCursor()
        output = PATHS.metrics_path.parent / "batch_results.csv"
        pd.DataFrame(csv_rows).to_csv(output, index=False, encoding="utf-8-sig")
        self.batch_table.setRowCount(len(preview_rows))
        for row_index, row in enumerate(preview_rows):
            for column_index, value in enumerate(row.values()):
                self.batch_table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        self.batch_table.resizeColumnsToContents()
        self.batch_status.setText(f"已处理 {len(preview_rows)} 张图像，结果已导出: {output}")

    def _build_stats_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        summary = QGroupBox("完整数据集统计")
        form = QFormLayout(summary)
        form.addRow("图像总数", QLabel(str(len(self.metadata))))
        form.addRow("鸟种类别", QLabel(str(self.metadata["class_id"].nunique())))
        form.addRow("训练集", QLabel(str((self.metadata["split"] == "train").sum())))
        form.addRow("测试集", QLabel(str((self.metadata["split"] == "test").sum())))
        form.addRow("随机种子", QLabel("2026"))
        layout.addWidget(summary)
        self.metrics_text = QPlainTextEdit()
        self.metrics_text.setReadOnly(True)
        if PATHS.metrics_path.is_file():
            metrics = json.loads(PATHS.metrics_path.read_text(encoding="utf-8"))
            regression = metrics.get("regression", {})
            rows = regression.get("official_split", {}).get("per_trait", [])
            regression_summary = "图像直接回归（官方测试集，原单位）\n" + "\n".join(
                f"{r['trait']}: MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}  R²={r['r2']:.4f}" for r in rows)
            self.metrics_text.setPlainText(regression_summary + "\n\n" + json.dumps(metrics, ensure_ascii=False, indent=2))
        else:
            self.metrics_text.setPlainText("尚未训练。请按 README 依次运行数据准备和训练评估脚本。")
        layout.addWidget(self.metrics_text)
        self.pca_label = QLabel("训练后会生成 PCA 分布图")
        self.pca_label.setAlignment(Qt.AlignCenter)
        figure = PATHS.metrics_path.parent / "regression_r2.png"
        if not figure.is_file():
            figure = PATHS.metrics_path.parent / "pca_species_features.png"
        if figure.is_file():
            self.pca_label.setPixmap(QPixmap(str(figure)).scaled(700, 430, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(self.pca_label)
        self.tabs.addTab(page, "统计与模型结果")

    def _build_help_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        help_text = QPlainTextEdit()
        help_text.setReadOnly(True)
        help_text.setPlainText(
            "系统用途\n\n"
            "本系统面向 CUB-200-2011 的 200 个鸟种。它采用 OpenCV 基础算子进行前景分割，"
            "提取颜色、形状、GLCM、LBP、分形和对称性特征，分别以 ExtraTrees 完成分类和 7 项形态性状回归。\n\n"
            "使用方式\n\n"
            "1. 在数据集浏览页检索并预览 CUB 图像。\n"
            "2. 在单图页查看直接回归值、分类结果及按预测鸟种查表的 AVONET 参考值。\n"
            "3. 在批量处理页选择文件夹并导出 CSV。\n\n"
            "适用范围\n\n"
            "外部图像只在 CUB-200 鸟种范围内进行预测。低置信度结果需要人工复核，"
            "回归值由图像特征独立预测，查表值由分类结果决定。二者均针对物种平均性状，不能替代个体实测。"
        )
        layout.addWidget(help_text)
        self.tabs.addTab(page, "程序说明")
