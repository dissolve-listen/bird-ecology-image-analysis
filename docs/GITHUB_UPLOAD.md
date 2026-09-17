# 一步一步上传 GitHub

本目录就是仓库根目录。上传整个项目中被 Git 跟踪的内容，不能只上传 `src`。已提供的 `.gitignore` 保留全部预处理数据和压缩模型，排除虚拟环境、原始压缩包、解压图像与复现临时输出。

1. 登录 https://github.com ，打开 https://github.com/new 。推荐仓库名 `bird-ecology-image-analysis`。课程要求开源时选择 Public。新仓库不要添加 README、.gitignore 或 License，因为本地已有文件。
2. 点击 Create repository，复制 HTTPS 地址，例如 `https://github.com/你的用户名/bird-ecology-image-analysis.git`。
3. 在本目录打开 PowerShell，用 `git status` 查看本地仓库。如果尚未初始化，执行 `git init -b main`、`git add .`、`git commit -m "Prepare reproducible coursework submission"`。若已经有提交则不要重复初始化。
4. 添加远端：`git remote add origin "刚复制的HTTPS地址"`。若已经有 origin，先用 `git remote -v` 核对，避免发往错误仓库。
5. 执行 `git push -u origin main`。如果弹出 Git Credential Manager 的浏览器登录窗口，使用自己的 GitHub 账号完成登录。不要把账号密码或令牌写进脚本、远端 URL 或提交文件。
6. 上传成功后刷新仓库，确认 `data/processed/image_features.csv`、两个 `models/*.joblib`、`requirements.lock` 和 `docs/` 都在。进入 Actions 查看 Verify reproducibility 工作流是否通过。
7. 在 Actions 中用 Run workflow 手动运行完整的“从预处理特征重新训练”校验。工作流不会下载 1.15 GB 原图；从原图重建按 README 在本机运行。
8. 再从远端克隆一份到新目录，按 README 安装依赖并运行 `scripts/verify_bundle.py`。需要核对重训练时运行 `scripts/reproduce.py --mode cached`。

两个模型已无损压缩，最大文件约 28.4 MiB。GitHub 网页单文件上传上限比普通 Git 小，所以请使用 Git 命令或 GitHub Desktop 提交整个仓库。当前包无需 Git LFS 或单独的 Release 模型附件。

官方上传教程：https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github 。

提交给老师的仓库地址使用 `https://github.com/你的用户名/仓库名`。纸质报告、电子报告和课堂展示仍按课程要求交付。
