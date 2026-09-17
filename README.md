# 鸟类生态图像分析与连续性状回归

数字图像处理综合实践课程项目。使用 OpenCV 基础算子分割前景，提取 68 维视觉特征，分别训练 ExtraTrees 鸟种分类器和七输出连续性状回归器。完整使用 CUB-200-2011 的 11,788 张图像、200 类及官方 5,994 / 5,794 训练测试划分，随机种子为 2026。

**回归器直接读取图像特征。预测鸟种和 AVONET 查表结果不会输入回归器。** 回归目标为体重、跗蹠长度、翅长、尾长、喙长、喙宽、喙深。监督标签是 AVONET 物种平均值，预测不能解释为照片中个体的实测值。

## 环境与首次安装

验证环境为 **Windows 64 位、Python 3.13.9**。请使用 Python 3.13；脚本会拒绝 3.14 等未验证的主次版本。`requirements.lock` 固定了直接和传递依赖，安装到本项目 `.venv`。不依赖作者的 Anaconda 路径或其他项目文件夹。

在克隆后的项目根目录打开 PowerShell：

```powershell
python --version
python -X utf8 scripts/bootstrap.py
.\.venv\Scripts\python.exe -X utf8 scripts/verify_bundle.py
```

如果 `python` 指向其他版本，请用已安装的 Python 3.13 可执行文件运行 `bootstrap.py`，例如 `py -3.13 -X utf8 scripts/bootstrap.py`。只需首次安装依赖时访问 Python 包索引。

## 直接启动展示

仓库已包含**两个经过无损压缩的完整训练模型**，无需另下权重。环境安装后双击 `启动界面.bat`，或运行：

```powershell
.\.venv\Scripts\python.exe run_app.py
```

未下载 CUB 原始图像时，可以打开自己的鸟类照片做单图和批量分析，也可以查看统计结果。数据集浏览中的原始照片需按下文准备 CUB。分类范围限于 CUB 的 200 种鸟，不能保证任意照片都能准确识别。

## 复现一：从随包预处理数据重新训练

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/reproduce.py --mode cached
```

这会校验输入文件，执行测试，用全部预处理特征**重新训练** 500 树分类器、500 树回归器、五折按鸟种分组验证和分类/回归消融，并与提交参考结果逐项核对。此流程不读取原始图片，不下载 AVONET，不调用分类查表代替回归。原始图像到特征的重建请使用下一种流程。

重建输出在 `reproduced/`，随包的 `models/`、`results/`、`reference/` 保持不变。全部核对通过才会生成 `reproduced/verification.json` 且 `passed: true`；失败返回非零退出码，详细日志在同一目录。再次失败运行会清除旧成功标记，不能把旧文件当成新验证结果。

## 复现二：从原始图像完整重建

本地已有官方压缩包时，把下例路径换成实际位置：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/reproduce.py --mode full --archive "E:\datasets\CUB_200_2011.tgz" --workers 4
```

没有压缩包时：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/reproduce.py --mode full --download --workers 4
```

完整流程下载或读取官方约 1.15 GB 的压缩包，核对 SHA-256，完整解压并逐张验证 11,788 张原始图像；用固定 AVONET 快照和映射重建标签；重新分割全部图像、提取全部特征、训练所有模型并评估。它同时比较重提特征、5,794 张测试预测、5,994 张训练集内组外预测、逐目标误差、消融、相关系数及最终指标。不是抽样验证。计算耗时与电脑性能有关，数据解压及环境需要数 GB 磁盘空间。

只准备数据以启用 GUI 数据集浏览：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/prepare_data.py --download
```

也可用一键脚本 `powershell -ExecutionPolicy Bypass -File .\完整构建.ps1 -Mode cached`；从图像重建使用 `-Mode full -Download`，已有压缩包则用 `-Archive "实际路径"`。可用 `-Python "Python 3.13 可执行文件路径"` 指定解释器。任何一步失败都会停止。

## 核对标准与已知范围

| 指标 | 提交参考值 |
|---|---:|
| 分类 Top-1 | 0.0866413531 |
| 分类 Top-5 | 0.2442181567 |
| 分类宏平均 F1 | 0.0705899928 |
| 分割平均目标框 IoU | 0.4571913941 |
| 七性状官方测试平均 R² | 0.1651881366 |
| 训练集内留鸟种五折平均 R² | 0.1084895729 |

原始文件按 SHA-256 核对；分类标签必须完全一致；数值比较使用 `rtol=1e-9, atol=1e-8`，容许浮点求和末位差异。模型二进制、图表字体或像素不作为重训一致性的判据。支持范围和实际运行记录见 [复现验证记录](docs/REPRODUCIBILITY.md)。本地已验证并不等于 GitHub Actions 已运行，上传后应检查 Actions 状态。

精度有限。官方训练测试包含相同鸟种，官方测试结果不代表未见鸟种泛化；按鸟种分组的组外验证只使用官方训练图像。真实类别仅用于关联监督标签；类别编号、文件名、预测类别、目标框及 IoU 不作为回归输入。CUB 官方目标框仅用于事后分割评价。

## 仓库内容

| 目录/文件 | 内容 |
|---|---|
| `src/`、`scripts/`、`tests/` | 程序、训练复现命令和测试 |
| `data/processed/` | 全部 11,788 张图像的特征、可迁移元数据、200 类性状标签 |
| `data/source/`、`resources/cub_to_scientific.csv` | 固定 AVONET 快照、CUB 官方标注、物种映射 |
| `data/manifest.json` | 数据来源、压缩包与全部图像校验值、输入/模型/参考文件校验值 |
| `models/` | 两个完整训练模型，均小于 GitHub 普通 Git 单文件限制 |
| `results/`、`reference/` | 已提交结果与自动验证使用的固定参考 |
| `docs/report/`、`docs/presentation/` | 正式课程报告、课堂 PPT、界面截图 |
| `.github/workflows/verify.yml` | 上传后自动运行的安装、数据/模型、测试和界面检查 |

原始图像和虚拟环境不入 Git；仓库保留预处理数据和模型。CUB 原始下载有官方链接与精确哈希，不依赖作者电脑。数据引用及第三方材料说明见 [DATA_SOURCES.md](docs/DATA_SOURCES.md)。报告重建可另装 `requirements-report.txt` 后执行 `scripts/build_report.py`；报告排版不是模型复现的必要步骤。课堂 PPT 已提供成品，不依赖作者专用的演示生成环境。

GitHub 上传步骤见 [GITHUB_UPLOAD.md](docs/GITHUB_UPLOAD.md)。课程要求中的纸质报告和课堂演示仍需另行完成。
