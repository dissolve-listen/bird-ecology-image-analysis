from __future__ import annotations

import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


AVONET_ARTICLE_URL = "https://figshare.com/ndownloader/articles/16586228/versions/7"
AVONET_LANDING_PAGE = "https://figshare.com/articles/dataset/AVONET_morphological_ecological_and_geographical_data_for_all_birds_Tobias_et_al_2021_Ecology_Letters_/16586228"
AVONET_BIRDTREE_MIRROR = "https://raw.githubusercontent.com/stephchia/bird-morphology-tda/main/data/raw/AVONET_birdtree.csv"

# CUB follows a more recent/common taxonomy for several North American birds.  The
# AVONET BirdTree table preserves older names, so these explicit synonyms keep the
# join auditable rather than relying on a fuzzy match. Spotted Catbird uses the
# closest pre-split AVONET taxon and is marked as a taxonomic proxy in the report.
TAXONOMIC_OVERRIDES = {
    "Spotted Catbird": "Ailuroedus crassirostris", "Chuck will Widow": "Caprimulgus carolinensis",
    "Brandt Cormorant": "Phalacrocorax penicillatus", "Red faced Cormorant": "Phalacrocorax urile",
    "Pelagic Cormorant": "Phalacrocorax pelagicus", "Purple Finch": "Carpodacus purpureus",
    "Gadwall": "Anas strepera", "American Goldfinch": "Carduelis tristis",
    "Evening Grosbeak": "Coccothraustes vespertinus", "Herring Gull": "Larus argentatus",
    "Green Violetear": "Colibri thalassinus", "Nighthawk": "Chordeiles minor",
    "Whip poor Will": "Caprimulgus vociferus", "Baird Sparrow": "Ammodramus bairdii",
    "Grasshopper Sparrow": "Ammodramus savannarum", "Henslow Sparrow": "Ammodramus henslowii",
    "Le Conte Sparrow": "Ammodramus leconteii", "Nelson Sharp tailed Sparrow": "Ammodramus nelsoni",
    "Seaside Sparrow": "Ammodramus maritimus", "Vesper Sparrow": "Pooecetes gramineus",
    "White crowned Sparrow": "Zonotrichia leucophrys", "Tree Swallow": "Tachycineta bicolor",
    "Summer Tanager": "Piranga rubra", "Artic Tern": "Sterna paradisaea", "Black Tern": "Chlidonias niger",
    "Caspian Tern": "Sterna caspia", "Elegant Tern": "Sterna elegans", "Forsters Tern": "Sterna forsteri",
    "Least Tern": "Sterna antillarum", "Green tailed Towhee": "Pipilo chlorurus", "Brown Thrasher": "Toxostoma rufum",
    "Black capped Vireo": "Vireo atricapilla", "Philadelphia Vireo": "Vireo philadelphicus",
    "Warbling Vireo": "Vireo gilvus", "White eyed Vireo": "Vireo griseus", "Yellow throated Vireo": "Vireo flavifrons",
    "Bay breasted Warbler": "Dendroica castanea", "Black throated Blue Warbler": "Dendroica caerulescens",
    "Blue winged Warbler": "Vermivora pinus", "Canada Warbler": "Wilsonia canadensis", "Cape May Warbler": "Dendroica tigrina",
    "Cerulean Warbler": "Dendroica cerulea", "Chestnut sided Warbler": "Dendroica pensylvanica",
    "Golden winged Warbler": "Vermivora chrysoptera", "Hooded Warbler": "Wilsonia citrina",
    "Kentucky Warbler": "Oporornis formosus", "Magnolia Warbler": "Dendroica magnolia",
    "Mourning Warbler": "Oporornis philadelphia", "Myrtle Warbler": "Dendroica coronata",
    "Nashville Warbler": "Vermivora ruficapilla", "Orange crowned Warbler": "Vermivora celata",
    "Palm Warbler": "Dendroica palmarum", "Pine Warbler": "Dendroica pinus", "Prairie Warbler": "Dendroica discolor",
    "Swainson Warbler": "Limnothlypis swainsonii", "Tennessee Warbler": "Vermivora peregrina",
    "Wilson Warbler": "Wilsonia pusilla", "Worm eating Warbler": "Helmitheros vermivorum",
    "Yellow Warbler": "Dendroica petechia", "Northern Waterthrush": "Seiurus noveboracensis",
    "Louisiana Waterthrush": "Seiurus motacilla", "American Three toed Woodpecker": "Picoides dorsalis",
    "Red cockaded Woodpecker": "Picoides borealis", "Red headed Woodpecker": "Melanerpes erythrocephalus",
    "Downy Woodpecker": "Picoides pubescens", "Bewick Wren": "Thryomanes bewickii",
    "Carolina Wren": "Thryothorus ludovicianus", "Marsh Wren": "Cistothorus palustris",
    "Rock Wren": "Salpinctes obsoletus", "Winter Wren": "Troglodytes troglodytes",
}


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z]", "", str(value).lower())


def display_to_query(display_name: str) -> str:
    return re.sub(r"\s+", " ", display_name.replace("_", " ").replace("-", " ")).strip()


def download_avonet(destination: Path, timeout: int = 90) -> Path:
    """Download the official AVONET archive and return its Supplementary Dataset 1 workbook."""
    destination.mkdir(parents=True, exist_ok=True)
    existing = list(destination.rglob("*Supplementary*dataset*1*.xlsx")) + list(destination.rglob("AVONET1_BirdLife.csv"))
    if existing:
        return existing[0]
    headers = {"User-Agent": "Mozilla/5.0 (compatible; CUB-Ecology-Coursework/1.0)", "Accept": "application/octet-stream,*/*"}
    response = requests.get(AVONET_ARTICLE_URL, headers=headers, timeout=timeout)
    if response.status_code == 200:
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        archive.extractall(destination)
        workbooks = list(destination.rglob("*Supplementary*dataset*1*.xlsx"))
        if workbooks:
            return workbooks[0]
    # The public Figshare endpoint occasionally rejects automated requests. This mirror preserves
    # the AVONET BirdTree table and lets course work remain reproducible without fabricating traits.
    mirror = requests.get(AVONET_BIRDTREE_MIRROR, headers=headers, timeout=timeout)
    if mirror.status_code == 200:
        fallback = destination / "AVONET_birdtree.csv"
        fallback.write_bytes(mirror.content)
        return fallback
    raise RuntimeError(
        "无法自动下载 AVONET 官方表或公开 BirdTree 镜像。请从以下页面下载 AVONET Supplementary dataset 1.xlsx，"
        f"放入 {destination} 后重试：{AVONET_LANDING_PAGE}"
    )


def load_avonet_table(source: Path) -> pd.DataFrame:
    if source.suffix.lower() == ".csv":
        table = pd.read_csv(source)
    else:
        table = pd.read_excel(source, sheet_name="AVONET1_BirdLife")
    table.columns = [str(column).strip() for column in table.columns]
    candidates = [column for column in ("Species1", "Species3", "Species", "Scientific.Name", "Scientific_name") if column in table.columns]
    if not candidates:
        raise ValueError(f"AVONET 表未找到物种列，现有列: {list(table.columns)[:20]}")
    table["scientific_name"] = table[candidates[0]].astype(str).str.strip()
    table["scientific_key"] = table["scientific_name"].map(normalize_name)
    return table


def resolve_scientific_name(common_name: str, session: requests.Session | None = None) -> str | None:
    """Resolve a CUB English common name through the public iNaturalist taxonomy API."""
    client = session or requests.Session()
    response = client.get(
        "https://api.inaturalist.org/v1/taxa/autocomplete",
        params={"q": display_to_query(common_name), "rank": "species", "per_page": 8},
        headers={"User-Agent": "CUB-Ecology-Coursework/1.0"},
        timeout=20,
    )
    response.raise_for_status()
    candidates = response.json().get("results", [])
    query_key = normalize_name(common_name)
    for candidate in candidates:
        if candidate.get("rank") != "species":
            continue
        common = normalize_name(candidate.get("preferred_common_name") or candidate.get("english_common_name") or "")
        if common == query_key or query_key in common or common in query_key:
            return candidate.get("name")
    for candidate in candidates:
        if candidate.get("rank") == "species" and candidate.get("name"):
            return candidate["name"]
    return None


def build_crosswalk(metadata: pd.DataFrame, avonet: pd.DataFrame, output_csv: Path) -> pd.DataFrame:
    """Create and validate one CUB class to AVONET species mapping per category."""
    classes = metadata[["class_id", "display_name"]].drop_duplicates().sort_values("class_id").copy()
    if output_csv.is_file():
        existing = pd.read_csv(output_csv)
        if len(existing) == len(classes) and set(existing["class_id"]) == set(classes["class_id"]):
            classes = classes.merge(existing[["class_id", "scientific_name"]], on="class_id", how="left")
        else:
            classes["scientific_name"] = None
    else:
        classes["scientific_name"] = None
    for display_name, scientific_name in TAXONOMIC_OVERRIDES.items():
        classes.loc[classes["display_name"] == display_name, "scientific_name"] = scientific_name
    missing = [(index, row["display_name"]) for index, row in classes.loc[classes["scientific_name"].isna()].iterrows()]
    if missing:
        print(f"正在通过 iNaturalist 解析 {len(missing)} 个 CUB 英文鸟种名...", flush=True)
        # Each task owns a requests session. Parallel requests reduce the first-run lookup from
        # several minutes to seconds while keeping the public API request volume modest.
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_index = {executor.submit(resolve_scientific_name, name): index for index, name in missing}
            resolved = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    classes.at[index, "scientific_name"] = future.result()
                except requests.RequestException:
                    classes.at[index, "scientific_name"] = None
                resolved += 1
                if resolved % 25 == 0 or resolved == len(missing):
                    print(f"已解析 {resolved}/{len(missing)} 个鸟种名", flush=True)
    classes["scientific_key"] = classes["scientific_name"].fillna("").map(normalize_name)
    direct = classes.merge(avonet, on="scientific_key", how="left", suffixes=("", "_avonet"))
    unresolved = direct[direct["scientific_name_avonet"].isna()][["class_id", "display_name", "scientific_name"]]
    if not unresolved.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        classes[["class_id", "display_name", "scientific_name"]].to_csv(output_csv, index=False, encoding="utf-8-sig")
        raise RuntimeError(
            "以下 CUB 类无法与 AVONET 自动关联，请在 resources/cub_to_scientific.csv 中确认科学名后重试: "
            + "; ".join(f"{r.display_name} ({r.scientific_name or '未解析'})" for r in unresolved.itertuples())
        )
    classes[["class_id", "display_name", "scientific_name"]].to_csv(output_csv, index=False, encoding="utf-8-sig")
    return direct


def select_traits(joined: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "class_id", "display_name", "scientific_name", "Mass", "Tarsus.Length", "Wing.Length", "Tail.Length",
        "Bill.Length", "Beak.Length_Culmen", "Bill.Width", "Beak.Width", "Bill.Depth", "Beak.Depth", "Primary.Lifestyle", "Primary.Diet", "Trophic.Niche", "Habitat", "Range.Size",
    ]
    columns = [column for column in wanted if column in joined.columns]
    traits = joined[columns].copy()
    rename = {
        "display_name": "鸟种", "scientific_name": "科学名", "Mass": "体重_g", "Tarsus.Length": "跗蹠长度_mm",
        "Wing.Length": "翅长_mm", "Tail.Length": "尾长_mm", "Bill.Length": "喙长_mm", "Bill.Width": "喙宽_mm",
        "Bill.Length": "喙长_mm", "Beak.Length_Culmen": "喙长_mm", "Bill.Width": "喙宽_mm", "Beak.Width": "喙宽_mm",
        "Bill.Depth": "喙深_mm", "Beak.Depth": "喙深_mm", "Primary.Lifestyle": "主要生活方式", "Primary.Diet": "主要食性", "Trophic.Niche": "主要食性",
        "Habitat": "栖息地", "Range.Size": "分布范围_km2",
    }
    return traits.rename(columns=rename)
