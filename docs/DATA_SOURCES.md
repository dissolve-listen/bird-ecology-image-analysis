# 数据来源与固定版本

## CUB-200-2011

作者：Catherine Wah、Steve Branson、Peter Welinder、Pietro Perona、Serge Belongie。文献：*The Caltech-UCSD Birds-200-2011 Dataset*, 2011, CNS-TR-2011-001。

- 官方数据页面：https://data.caltech.edu/records/65de6-vp158
- DOI：https://doi.org/10.22002/D1.20098
- 下载：https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1
- 官方页面 MD5：`97eceeb196236b17998738112f37df78`。
- 本项目归档 SHA-256：`0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081`。
- 文件大小：1,150,585,339 字节。200 类、11,788 图像，官方训练 5,994、测试 5,794。

`data/source/cub_annotations/` 保留构建元数据所需的官方标注及原始 README。`data/processed/image_features.csv` 为本项目基础算子生成的全部特征，而非原始照片。原始照片从官方页面获取。图像及第三方材料的权利归原权利人；本仓库的课程代码不重新授予这些材料的使用权。课程报告与 PPT 含用于实验说明的图像。

## AVONET

Tobias et al. (2022), *AVONET: morphological, ecological and geographical data for all birds*, Ecology Letters。官方数据入口：https://figshare.com/articles/dataset/16586228 。AVONET 数据按原始 CC BY 4.0 条款归属原作者；许可证说明：https://creativecommons.org/licenses/by/4.0/ 。

本实验实际使用 BirdTree 分类体系 CSV。随包 `data/source/AVONET_birdtree.csv` 是本次计算使用的固定快照（2,205,150 字节），取得自公开镜像：https://raw.githubusercontent.com/stephchia/bird-morphology-tda/main/data/raw/AVONET_birdtree.csv 。该可变链接仅作来源记录，复现流程读取本仓库固定文件，并以 `data/manifest.json` 的 SHA-256 验证，不在每次运行时重新下载最新数据。

`resources/cub_to_scientific.csv` 固定了全部 200 类英文名与学名的关联。Vesper Sparrow 对应 `Pooecetes gramineus`，不与 Savannah Sparrow 共享错误学名。核对来源：https://www.allaboutbirds.org/guide/Vesper_Sparrow/overview 。Spotted Catbird 使用 AVONET 拆分前近缘分类单元 `Ailuroedus crassirostris` 作为分类代理，须在解释结果时保留此限制。

七项形态性状在全部 200 类中完整且为正值，回归不删除照片、不填造标签。分布范围存在缺失，仅作查表信息。标签是物种平均值，同物种不同照片使用相同监督标签；测试时回归器仍只接收 68 维视觉特征。
