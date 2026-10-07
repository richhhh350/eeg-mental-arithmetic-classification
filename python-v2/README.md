# Python V2：可追溯 EEG 静息/心算分类

独立于原 MATLAB V1 的实现。第一阶段已完成 36 折嵌套 LOSO 逻辑回归基线，详见[阶段报告](../docs/v2-stage1-report.md)。EDF、环境、模型及完整中间结果不随仓库分发。

## 安装

使用 **Python 3.12**。Windows PowerShell 中，进入本目录后运行：

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
& .\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
& .\.venv\Scripts\python.exe -m pytest -q
```

若没有 `py` 启动器，将首行替换为已安装 Python 3.12 可执行文件的完整路径。项目依赖在 `.venv` 内安装；锁文件记录本次 Windows/Python 3.12.14 实际使用的版本，其他平台的数值一致性需另行验证。

本机默认 pytest 临时目录受权限限制，本次使用项目内的新临时目录并保存报告：

```powershell
& .\.venv\Scripts\python.exe -m pytest -q --basetemp=outputs/test-temp-new --junitxml=outputs/tests.xml
```

每次选择新的 `--basetemp` 路径；pytest 会清理它接管的临时目录，不能指向需要保留的实验数据。

## 数据与命令

从 [PhysioNet EEGMAT 1.0.0](https://physionet.org/content/eegmat/1.0.0/) 单独获取数据。`--data-dir` 必须直接包含 `Subject00_1.edf` 至 `Subject35_2.edf` 共 72 份 EDF。本地原项目数据目录相对于此文件夹为：

```powershell
$eegDataDir = '..\..\..\eeg-during-mental-arithmetic-tasks-1.0.0\eeg-during-mental-arithmetic-tasks-1.0.0'
& .\.venv\Scripts\python.exe -m eeg_v2.cli prepare --data-dir $eegDataDir --config configs/stage1.json --output outputs/prepared-new
& .\.venv\Scripts\python.exe -m eeg_v2.cli run --prepared outputs/prepared-new --output outputs/smoke-new --subjects Subject00 Subject17 Subject35
& .\.venv\Scripts\python.exe -m eeg_v2.cli run --prepared outputs/prepared-new --output outputs/loso-new
& .\.venv\Scripts\python.exe -m eeg_v2.cli verify --output outputs/loso-new
```

克隆仓库后，将 `$eegDataDir` 改为自己的数据解压目录。只复现一折时在 `run` 后加 `--subjects Subject00`。准备数据和训练均要求不存在的新输出目录，避免覆盖已有证据。改变特征配置后应重新 `prepare`，不能直接挪用旧特征；`run` 会验证输入产物 hash。

本次运行保存在本地 `outputs/prepared-stage1`、`outputs/smoke-stage1`、`outputs/loso-stage1`。上述命令使用不同的目录名称，可另行复现而不覆盖当前结果。

## 冻结协议

配置文件是 [configs/stage1.json](configs/stage1.json)。

- 任务：静息=0、心算=1，36 名被试，19 导联，500 Hz，µV。
- 裁切：每记录前 60 秒。所有 EDF 注释无文字任务事件，实际任务起点无法确认；这是一条固定对接规则。
- 处理：当前窗口内平均参考和去均值；不新增时间滤波或 ICA。供应方已清洗的数据边界不能撤销或重新认证。
- 窗口：2 秒、无重叠；DE 使用 θ 4—8、α 8—13、β 13—30 Hz，频段端点闭合，输出 57 维，按通道优先、频段其次排列。
- S2：外层按人留一，源侧 3 折 GroupKFold；C 候选为 0.01、0.1、1、10，按验证被试 BA 等权均值选优，平局取小 C。标准化每次仅拟合训练数据（ddof=0）。
- 模型决策阈值固定 0.5。测试标签不参与选择、标准化或拟合。
- S1：先按原始时间划块，再切窗口，留 4 秒间隔；本阶段只提供时间清单与隔离测试，尚未训练 S1 模型。新增时间滤波后必须重新计算间隔与裁边。
- 失败：记录异常与失败状态，不能用 Dummy 预测掩盖失败；未完成全部请求折时不输出完整平均成绩。

V1 与 V2 的预处理、频段、重叠和选参同时不同。当前数值不能直接解释为某一改动导致的性能增益；尚未实施 EEGNet、CORAL 或 DANN。

## 文件与证据

| 文件 | 用途 |
|---|---|
| `src/eeg_v2/data.py` | EDF 匿名元数据、SHA-256、按标签加载、描述性质量检查 |
| `features.py` | V1 兼容功率、DE、LEE 公式；没有隐式预处理 |
| `splits.py` | 半开时间区间、完整窗口、跨 split 交集断言 |
| `evaluation.py` | 源侧选参与拟合、被试级指标 |
| `cli.py` | 数据准备、运行、保存预测复算与图表 |
| `outputs/prepared-*/windows.csv` | 原记录采样区间、条件、窗口及处理哈希 |
| `s1_windows.csv` | 训练/验证/测试块和窗口边界 |
| `features.npz` | 唯一进入模型的数值特征，以及分离的标签/分组向量 |
| `manifest.json` | 配置、输入文件、源码、环境及产物指纹 |
| `quality.json` / `records.json` | 全记录质量指标及匿名来源摘要 |
| `outputs/loso-*/Subject*_selection.json` | 候选分数、内层参与者、最终拟合/测试窗口清单 |
| `Subject*_model.joblib` | 标准化与 LR；仅加载可信的本地模型文件 |
| `predictions.csv` / `subject_metrics.csv` | 逐窗概率和逐被试指标 |
| `run.json` / `summary.json` / `verification.json` | 运行来源、汇总和独立复算检查 |

不通过行数、通道数或窗口数人为放大被试级推断样本量。质量异常标志仅是描述性启发式，不等于完整伪迹诊断。

## MATLAB 同输入对照（可选）

此检查需要原 V1 预处理 MAT 和已授权可运行的 MATLAB；普通 Python 基线不需要 MATLAB。脚本只读原数据，结果写入新目录：

```powershell
& .\.venv\Scripts\python.exe scripts/validate_legacy.py --data-dir $eegDataDir --legacy-dir '..\..\..\preprocessed_data' --output-dir outputs/legacy-check-new --matlab 'C:\Program Files\MATLAB\R2024a\bin\matlab.exe'
```

检查相同实测输入的功率/DE，并核对旧 MAT 的采样数和 epoch 边界。未安装 MATLAB 时明确记录不可用，不会把未运行计为成功。当前已完成对照证据见[阶段报告](../docs/v2-stage1-report.md)。
