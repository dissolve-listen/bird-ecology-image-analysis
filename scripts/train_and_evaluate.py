from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler

from src.config import PATHS, SEED
from src.dataset import load_cub_metadata, validate_cub_metadata, validate_feature_cache, read_portable_metadata
from src.features import extract_features
from src.modeling import ablation_columns, make_classifier, save_json, save_model, train_model
from src.regression import evaluate_regression


def iou_xywh(predicted: tuple[int, int, int, int], ground_truth: tuple[float, float, float, float]) -> float:
    px, py, pw, ph = predicted
    gx, gy, gw, gh = ground_truth
    left, top = max(px, gx), max(py, gy)
    right, bottom = min(px + pw, gx + gw), min(py + ph, gy + gh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = pw * ph + gw * gh - intersection
    return float(intersection / union) if union else 0.0


def _read_image(path: str) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def extract_one(row: dict) -> dict:
    image = _read_image(row["image_path"])
    if image is None:
        return {"image_id": row["image_id"], "error": "image_read_failed"}
    feature_map, segmentation = extract_features(image)
    output = {
        "image_id": int(row["image_id"]),
        "class_id": int(row["class_id"]),
        "class_name": row["class_name"],
        "display_name": row["display_name"],
        "split": row["split"],
        "relative_path": row["relative_path"],
        "pred_bbox_x": segmentation.bbox[0],
        "pred_bbox_y": segmentation.bbox[1],
        "pred_bbox_width": segmentation.bbox[2],
        "pred_bbox_height": segmentation.bbox[3],
        "bbox_iou": iou_xywh(segmentation.bbox, (row["bbox_x"], row["bbox_y"], row["bbox_width"], row["bbox_height"])),
    }
    output.update(feature_map)
    return output


def build_features(metadata: pd.DataFrame, workers: int, force: bool) -> pd.DataFrame:
    if PATHS.features_csv.is_file() and not force:
        cached = pd.read_csv(PATHS.features_csv)
        validate_feature_cache(cached, metadata)
        print(f"使用已有全量特征缓存: {PATHS.features_csv}")
        return cached
    records = metadata.to_dict("records")
    print(f"开始提取 {len(records)} 张完整 CUB 图像的特征，工作线程: {workers}")
    extracted = Parallel(n_jobs=workers, prefer="processes", batch_size=20)(delayed(extract_one)(record) for record in records)
    failures = [row for row in extracted if "error" in row]
    if failures:
        raise RuntimeError(f"共有 {len(failures)} 张图像无法提取特征，例如: {failures[:3]}")
    features = pd.DataFrame(extracted).sort_values("image_id")
    if len(features) != 11788:
        raise RuntimeError(f"特征记录数异常: {len(features)}")
    PATHS.features_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(PATHS.features_csv, index=False, encoding="utf-8-sig")
    return features


def save_figures(features: pd.DataFrame, feature_columns: list[str], model) -> None:
    results = PATHS.metrics_path.parent
    test = features.loc[features["split"] == "test"]
    matrix = confusion_matrix(test["class_id"], test["predicted_class_id"], labels=sorted(test["class_id"].unique()))
    plt.figure(figsize=(13, 11))
    plt.imshow(matrix, cmap="Blues", interpolation="nearest")
    plt.colorbar(label="图像数量")
    plt.title("CUB-200 测试集混淆矩阵")
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.tight_layout()
    plt.savefig(results / "confusion_matrix.png", dpi=180)
    plt.close()

    means = features.groupby(["class_id", "display_name"])[feature_columns].mean().reset_index()
    scaled = StandardScaler().fit_transform(means[feature_columns])
    embedding = PCA(n_components=2, random_state=SEED).fit_transform(scaled)
    plt.figure(figsize=(11, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=means["class_id"], cmap="turbo", s=22, alpha=0.85)
    plt.title("200 个鸟种的特征 PCA 分布")
    plt.xlabel("主成分 1")
    plt.ylabel("主成分 2")
    plt.colorbar(label="类别编号")
    plt.tight_layout()
    plt.savefig(results / "pca_species_features.png", dpi=180)
    plt.close()

    importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False).head(20).sort_values()
    plt.figure(figsize=(10, 7))
    importances.plot.barh(color="#2878B5")
    plt.title("前 20 个分类特征重要性")
    plt.xlabel("ExtraTrees importance")
    plt.tight_layout()
    plt.savefig(results / "feature_importance.png", dpi=180)
    plt.close()


def evaluate(features: pd.DataFrame, run_ablation: bool) -> dict:
    model, feature_columns = train_model(features, n_estimators=500)
    save_model(model, feature_columns, PATHS.model_path)
    test = features.loc[features["split"] == "test"].copy()
    probabilities = model.predict_proba(test[feature_columns])
    predicted = model.classes_[np.argmax(probabilities, axis=1)]
    test["predicted_class_id"] = predicted
    test.to_csv(PATHS.metrics_path.parent / "test_predictions.csv", index=False, encoding="utf-8-sig")
    labels = sorted(features["class_id"].unique())
    metrics = {
        "seed": SEED,
        "dataset": {"images": int(len(features)), "classes": int(features["class_id"].nunique()), "train": int((features["split"] == "train").sum()), "test": int(len(test))},
        "classifier": {"name": "ExtraTreesClassifier", "n_estimators": 500, "max_features": "sqrt", "class_weight": "balanced", "max_leaf_nodes": 256, "min_samples_leaf": 2},
        "classification": {
            "top1_accuracy": float(accuracy_score(test["class_id"], predicted)),
            "top5_accuracy": float(top_k_accuracy_score(test["class_id"], probabilities, labels=model.classes_, k=5)),
            "macro_f1": float(f1_score(test["class_id"], predicted, average="macro", zero_division=0)),
        },
        "segmentation": {
            "mean_bbox_iou": float(test["bbox_iou"].mean()),
            "iou_at_0_5": float((test["bbox_iou"] >= 0.5).mean()),
            "mean_mask_quality": float(test["segmentation_quality"].mean()),
            "note": "CUB 官方目标框只用于事后评价，未输入分割器。",
        },
    }
    report = classification_report(test["class_id"], predicted, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(PATHS.metrics_path.parent / "classification_report.csv", encoding="utf-8-sig")
    if run_ablation:
        rows = []
        for name, columns in ablation_columns(feature_columns).items():
            candidate = make_classifier(n_estimators=150)
            candidate.fit(features.loc[features["split"] == "train", columns], features.loc[features["split"] == "train", "class_id"])
            candidate_predicted = candidate.predict(test[columns])
            rows.append({"特征组": name, "特征数": len(columns), "树数量": 150, "Top1准确率": accuracy_score(test["class_id"], candidate_predicted), "Macro_F1": f1_score(test["class_id"], candidate_predicted, average="macro", zero_division=0)})
        ablation = pd.DataFrame(rows)
        ablation.to_csv(PATHS.metrics_path.parent / "ablation_results.csv", index=False, encoding="utf-8-sig")
        metrics["ablation"] = ablation.to_dict(orient="records")
    save_figures(test, feature_columns, model)
    save_json(metrics, PATHS.metrics_path)
    return metrics


def main() -> None:
    global PATHS
    parser = argparse.ArgumentParser(description="全量训练并评估鸟类生态图像评价系统")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--regression-only", action="store_true", help="复用完整图像特征，只补充回归训练与评估，保留已有分类结果")
    parser.add_argument("--cached-features", action="store_true", help="用随包完整特征训练，不依赖原始图片")
    parser.add_argument("--output-dir", type=Path, help="把重建模型和结果写入独立目录，保留提交参考文件")
    args = parser.parse_args()
    if args.cached_features and args.force_features:
        parser.error("--cached-features 与 --force-features 不能同时使用")
    if args.output_dir:
        output = args.output_dir.resolve()
        PATHS = replace(PATHS, model_path=output / "models/extra_trees_full.joblib",
                        regression_model_path=output / "models/extra_trees_traits.joblib",
                        metrics_path=output / "results/metrics.json",
                        features_csv=(output / "data/processed/image_features.csv") if args.force_features else PATHS.features_csv)
    PATHS.make_output_dirs()
    if args.cached_features:
        metadata = read_portable_metadata(PATHS.metadata_csv, PATHS.cub_root)
        features = pd.read_csv(PATHS.features_csv)
    else:
        metadata = load_cub_metadata(PATHS.cub_root)
        print("数据完整性:", validate_cub_metadata(metadata, PATHS.cub_root))
        features = build_features(metadata, args.workers, args.force_features)
    validate_feature_cache(features, metadata)
    if args.regression_only:
        import json
        metrics = json.loads(PATHS.metrics_path.read_text(encoding="utf-8")) if PATHS.metrics_path.is_file() else {}
    else:
        metrics = evaluate(features, run_ablation=not args.skip_ablation)
    metrics["regression"] = evaluate_regression(features, pd.read_csv(PATHS.traits_csv),
        PATHS.regression_model_path, PATHS.metrics_path.parent, run_ablation=not args.skip_ablation)
    save_json(metrics, PATHS.metrics_path)
    print(json_dumps(metrics))


def json_dumps(value: dict) -> str:
    import json
    return json.dumps(value, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
