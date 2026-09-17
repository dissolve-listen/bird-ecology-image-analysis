from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SEED = 2026
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESOURCES_DIR = PROJECT_ROOT / "resources"


@dataclass(frozen=True)
class AppPaths:
    project_root: Path = PROJECT_ROOT
    cub_root: Path = RAW_DIR / "CUB_200_2011"
    metadata_csv: Path = PROCESSED_DIR / "cub_metadata.csv"
    features_csv: Path = PROCESSED_DIR / "image_features.csv"
    traits_csv: Path = PROCESSED_DIR / "cub_avonet_traits.csv"
    crosswalk_csv: Path = RESOURCES_DIR / "cub_to_scientific.csv"
    model_path: Path = MODELS_DIR / "extra_trees_full.joblib"
    regression_model_path: Path = MODELS_DIR / "extra_trees_traits.joblib"
    metrics_path: Path = RESULTS_DIR / "metrics.json"

    def make_output_dirs(self) -> None:
        for directory in (self.cub_root.parent, self.metadata_csv.parent,
                          self.features_csv.parent, self.model_path.parent,
                          self.regression_model_path.parent, self.metrics_path.parent,
                          self.crosswalk_csv.parent):
            directory.mkdir(parents=True, exist_ok=True)


PATHS = AppPaths()


def find_cub_archive(explicit: str | None = None) -> Path:
    """Locate the supplied CUB archive without copying it into the project."""
    candidates = []
    if explicit:
        archive = Path(explicit).expanduser().resolve()
        if not archive.is_file():
            raise FileNotFoundError(f"指定的 CUB 压缩包不存在: {archive}")
        return archive
    candidates.extend(
        [
            PROJECT_ROOT.parent / "CUB_200_2011.tgz",
            PROJECT_ROOT / "CUB_200_2011.tgz",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "未找到 CUB_200_2011.tgz。请把原始压缩包放到项目上级目录，或通过 --archive 指定路径。"
    )
