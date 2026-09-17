"""Direct image-to-AVONET regression, separate from species lookup.

The labels are species means. Official image holdout scores measure prediction
of new photographs of known species. Train-only grouped OOF scores separately
test transfer to unseen species; those diagnostic models are never deployed.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .config import SEED
from .modeling import ablation_columns, image_feature_columns, save_json

TRAIT_COLUMNS = ["体重_g", "跗蹠长度_mm", "翅长_mm", "尾长_mm", "喙长_mm", "喙宽_mm", "喙深_mm"]
TRAIT_NOTE = "由图像特征直接回归 AVONET 物种平均形态性状，不代表照片中个体的实测值。"


def regression_data(features: pd.DataFrame, traits: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    columns = image_feature_columns(features)
    if features["image_id"].duplicated().any() or features["image_id"].isna().any():
        raise ValueError("图像 ID 缺失或重复，不能可靠划分回归数据。")
    if set(features["split"].unique()) != {"train", "test"}:
        raise ValueError("回归需要明确的官方 train/test 划分。")
    missing = set(["class_id", *TRAIT_COLUMNS]) - set(traits.columns)
    if missing:
        raise ValueError(f"性状表缺少回归目标列: {sorted(missing)}")
    if traits["class_id"].duplicated().any():
        raise ValueError("每个 class_id 必须且只能对应一条性状记录。")
    if "科学名" in traits:
        names = traits.loc[traits["class_id"].isin(features["class_id"]), "科学名"].astype("string").str.strip().str.casefold()
        if names.isna().any() or names.eq("").any() or names.duplicated().any():
            raise ValueError("科学名缺失或多个 CUB 类共享同一 AVONET 学名；请核对映射，避免留鸟种验证的标签泄漏。")
    # class_id joins labels only. It is never passed to a regressor.
    joined = features.merge(traits[["class_id", *TRAIT_COLUMNS]], on="class_id", how="left", validate="many_to_one")
    y = joined[TRAIT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid = ~np.isfinite(y.to_numpy()) | (y.to_numpy() <= 0)
    if invalid.any():
        bad = joined.loc[invalid.any(axis=1), "class_id"].unique().tolist()
        raise ValueError(f"形态性状缺失、非数值或非正数，未删除图像或填造标签；请核对类别 {bad}")
    joined[TRAIT_COLUMNS] = y
    return joined, columns


def make_regressor(n_estimators: int = 500) -> TransformedTargetRegressor:
    # Fit target scaling ONLY on training labels to stop grams from dominating
    # the multioutput split criterion. Inverse transform restores g/mm units.
    return TransformedTargetRegressor(
        regressor=ExtraTreesRegressor(n_estimators=n_estimators, max_features=1.0,
            max_leaf_nodes=256, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        transformer=StandardScaler(),
    )


def regression_scores(actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray) -> list[dict]:
    rows = []
    for index, target in enumerate(TRAIT_COLUMNS):
        y, p, b = actual[:, index], predicted[:, index], baseline[:, index]
        # Undefined R² is null, never fabricated as a perfect score.
        r2 = float(r2_score(y, p)) if len(y) > 1 and np.ptp(y) > 0 else None
        baseline_r2 = float(r2_score(y, b)) if len(y) > 1 and np.ptp(y) > 0 else None
        rows.append({"trait": target, "unit": target.rsplit("_", 1)[1], "n": len(y),
            "mae": float(mean_absolute_error(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": r2, "baseline_mae": float(mean_absolute_error(y, b)),
            "baseline_rmse": float(np.sqrt(mean_squared_error(y, b))), "baseline_r2": baseline_r2})
    return rows


def prediction_table(data: pd.DataFrame, predicted: np.ndarray, baseline: np.ndarray) -> pd.DataFrame:
    result = data[["image_id", "class_id", "split", "relative_path"]].copy()
    for index, target in enumerate(TRAIT_COLUMNS):
        result[f"true_{target}"] = data[target].to_numpy()
        result[f"pred_{target}"] = predicted[:, index]
        result[f"baseline_{target}"] = baseline[:, index]
    return result


def save_regressor(model, columns: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": columns, "target_columns": TRAIT_COLUMNS,
        "seed": SEED, "target_transform": "StandardScaler fitted on official train labels only",
        "label_scope": "AVONET species means", "schema_version": 1}, path, compress=3)


def load_regressor(path: Path) -> tuple[TransformedTargetRegressor, list[str], list[str]]:
    artifact = joblib.load(path)
    return artifact["model"], artifact["feature_columns"], artifact["target_columns"]


def evaluate_regression(features: pd.DataFrame, traits: pd.DataFrame, model_path: Path,
                        results_dir: Path, n_estimators: int = 500,
                        group_folds: int = 5, run_ablation: bool = True) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    data, columns = regression_data(features, traits)
    train = data.loc[data["split"] == "train"].reset_index(drop=True)
    test = data.loc[data["split"] == "test"].reset_index(drop=True)
    if group_folds < 2 or group_folds > train["class_id"].nunique():
        raise ValueError("分组交叉验证折数应在 2 与训练鸟种数量之间。")
    x_train, y_train = train[columns], train[TRAIT_COLUMNS].to_numpy()
    print(f"直接回归: {len(train)} 训练 / {len(test)} 测试，{len(columns)} 维图像特征，7 项性状", flush=True)
    model = make_regressor(n_estimators)
    model.fit(x_train, y_train)
    predicted = model.predict(test[columns])
    baseline = np.broadcast_to(y_train.mean(axis=0), predicted.shape).copy()
    scores = regression_scores(test[TRAIT_COLUMNS].to_numpy(), predicted, baseline)
    predictions = prediction_table(test, predicted, baseline)
    save_regressor(model, columns, model_path)
    predictions.to_csv(results_dir / "regression_test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(scores).to_csv(results_dir / "regression_metrics.csv", index=False, encoding="utf-8-sig")

    # Cross-species diagnostic uses official TRAIN photographs only. Thus no
    # official test photo can affect the deployed model or any preprocessing.
    oof, oof_baseline = np.empty_like(y_train), np.empty_like(y_train)
    fold_ids = np.zeros(len(train), dtype=int)
    fold_details = []
    for fold, (fit_idx, val_idx) in enumerate(GroupKFold(group_folds).split(x_train, groups=train["class_id"]), 1):
        fit_species = sorted(train.iloc[fit_idx]["class_id"].unique().tolist())
        val_species = sorted(train.iloc[val_idx]["class_id"].unique().tolist())
        assert not set(fit_species) & set(val_species)
        print(f"训练集内留鸟种验证 {fold}/{group_folds}，留出 {len(val_species)} 类", flush=True)
        candidate = make_regressor(n_estimators)
        candidate.fit(x_train.iloc[fit_idx], y_train[fit_idx])
        oof[val_idx] = candidate.predict(x_train.iloc[val_idx])
        oof_baseline[val_idx] = y_train[fit_idx].mean(axis=0)
        fold_ids[val_idx] = fold
        fold_details.append({"fold": fold, "train_classes": fit_species, "validation_classes": val_species,
                             "train_images": len(fit_idx), "validation_images": len(val_idx)})
    group_scores = regression_scores(y_train, oof, oof_baseline)
    group_predictions = prediction_table(train, oof, oof_baseline)
    group_predictions["fold"] = fold_ids
    group_predictions.to_csv(results_dir / "regression_group_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(group_scores).to_csv(results_dir / "regression_group_metrics.csv", index=False, encoding="utf-8-sig")

    ablation_rows = []
    if run_ablation:
        for name, selected in ablation_columns(columns).items():
            candidate = make_regressor(150)
            candidate.fit(train[selected], y_train)
            rows = regression_scores(test[TRAIT_COLUMNS].to_numpy(), candidate.predict(test[selected]), baseline)
            ablation_rows.extend({"feature_group": name, "feature_count": len(selected), "n_estimators": 150, **row} for row in rows)
        pd.DataFrame(ablation_rows).to_csv(results_dir / "regression_ablation_results.csv", index=False, encoding="utf-8-sig")

    # Correlations use one mean feature vector per TRAIN species, not repeated
    # species labels treated as independent ecological observations.
    means = train.groupby("class_id")[[*columns, *TRAIT_COLUMNS]].mean()
    correlations = means.corr(method="spearman").loc[columns, TRAIT_COLUMNS]
    correlations.to_csv(results_dir / "regression_feature_trait_correlations.csv", encoding="utf-8-sig")
    metrics = {"seed": SEED, "model": "ExtraTreesRegressor + train-only target StandardScaler",
        "n_estimators": n_estimators, "max_features": 1.0, "max_leaf_nodes": 256, "min_samples_leaf": 2,
        "feature_count": len(columns), "targets": TRAIT_COLUMNS, "note": TRAIT_NOTE,
        "excluded_predictors": ["class_id", "image_id", "relative_path", "bbox_iou", "AVONET traits", "classifier predictions"],
        "official_split": {"train": len(train), "test": len(test), "classes": int(data["class_id"].nunique()),
            "scope": "官方图像留出，训练测试包含相同鸟种；不能据此声称未见鸟种泛化。", "per_trait": scores,
            "mean_r2": float(np.mean([row["r2"] for row in scores if row["r2"] is not None]))},
        "grouped_validation": {"source": "official train only", "folds": group_folds, "images": len(train),
            "scope": "按 class_id 分组的训练集内 OOF，每折训练与验证鸟种互斥；未用于调参或部署。",
            "per_trait": group_scores, "mean_r2": float(np.mean([row["r2"] for row in group_scores if row["r2"] is not None])),
            "fold_details": fold_details},
        "baseline": "各训练折按图像加权的目标均值，原单位评估", "ablation": ablation_rows}
    save_json(metrics, results_dir / "regression_metrics.json")
    save_regression_figures(test, predicted, scores, group_scores, correlations, results_dir)
    return metrics


def save_regression_figures(test, predicted, scores, group_scores, correlations, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for index, target in enumerate(TRAIT_COLUMNS):
        ax = axes.flat[index]
        actual = test[target].to_numpy()
        ax.scatter(actual, predicted[:, index], s=3, alpha=0.18, rasterized=True)
        low, high = min(actual.min(), predicted[:, index].min()), max(actual.max(), predicted[:, index].max())
        ax.plot([low, high], [low, high], "--", color="tomato", linewidth=1)
        ax.set(title=f"{target}  R²={scores[index]['r2']:.3f}", xlabel="AVONET 物种参考值", ylabel="图像回归值")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.05, 0.65, f"官方测试集 {len(test):,} 张图\n每点为一张图像\n虚线为理想预测\n标签是物种均值，非个体测量", fontsize=12)
    fig.tight_layout()
    fig.savefig(results_dir / "regression_predictions.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    indices = np.arange(len(TRAIT_COLUMNS))
    ax.bar(indices - 0.2, [r["r2"] for r in scores], 0.4, label="官方图像留出")
    ax.bar(indices + 0.2, [r["r2"] for r in group_scores], 0.4, label="训练集内留鸟种 OOF")
    ax.set_xticks(indices, [name.rsplit("_", 1)[0] for name in TRAIT_COLUMNS])
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set(ylabel="R²", title="图像性状回归的两种评价范围")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "regression_r2.png", dpi=180)
    plt.close(fig)
    top = correlations.abs().max(axis=1).nlargest(18).index
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(correlations.loc[top], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(top)), top)
    ax.set_xticks(range(len(TRAIT_COLUMNS)), [t.rsplit("_", 1)[0] for t in TRAIT_COLUMNS])
    ax.set_title("训练集 200 鸟种均值的 Spearman 相关（不表示因果）")
    fig.colorbar(im, ax=ax, label="Spearman ρ")
    fig.tight_layout()
    fig.savefig(results_dir / "regression_correlations.png", dpi=180)
    plt.close(fig)
