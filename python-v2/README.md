# Python V2：可追溯 EEG 静息/心算分类

**推荐完整版主线已完成（2026-10-09）**：见[总报告](../docs/v2-final-report.md)和[第四阶段报告](../docs/v2-stage4-report.md)。第四阶段252个有限校准组合、999次整被试条件交换、3组来源拟合UMAP及108模型的540次脑区扰动全部完成并审计，162项测试通过。DANN、额外ICA与外部数据验证仍为选做扩展。

独立于原 MATLAB V1 的实现。第一阶段已完成 36 折嵌套 LOSO 逻辑回归基线，详见[阶段报告](../docs/v2-stage1-report.md)。EDF、环境、模型及完整中间结果不随仓库分发。

**第二阶段（2026-10-08）**已完成 S1 对照、传统模型比较、单因素消融与来源训练 PCA，见[第二阶段报告](../docs/v2-stage2-report.md)。第一阶段命令和产物保持可用。

**第三阶段（2026-10-09）**已完成500→128 Hz窗口内抗混叠、五组频率网格/模型桥接，以及36人×种子13/42/2026的全部108个EEGNet组合。共288个模型、17,280条预测通过独立审计，模型重放最大误差为0；148项测试通过。EEGNet三种子指标均值BA为66.84%、macro-F1为0.6231、AUC为0.7631。配对区间均跨零，比较保持探索性，见[第三阶段报告](../docs/v2-stage3-report.md)和[实验协议](../docs/v2-stage3-protocol.md)。

## 第四阶段复现

先在新的工作副本完成第三阶段准备、训练和审计；脑区解释需要全部108个本地模型。以下命令在 `python-v2` 执行。第四阶段代码放在 `src/eeg_v2/stage4/` 子包，前三阶段顶层源码和冻结配置未改动。新增UMAP依赖锁定在 `requirements-stage4-lock.txt`。

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-stage4-lock.txt
& .\.venv\Scripts\python.exe -m pytest -q --basetemp=outputs/stage4-test-temp-new --junitxml=outputs/stage4-tests.xml
& .\.venv\Scripts\python.exe -m eeg_v2.stage4.runner run
& .\.venv\Scripts\python.exe -m eeg_v2.stage4.runner audit
& .\.venv\Scripts\python.exe -m eeg_v2.stage4.permutation --workers 4
& .\.venv\Scripts\python.exe -m eeg_v2.stage4.analysis
& .\.venv\Scripts\python.exe scripts/audit_stage4.py
& .\.venv\Scripts\python.exe scripts/report_stage4.py
```

默认输出为 `outputs/runs-stage4`、`outputs/permutation-stage4`、`outputs/analysis-stage4`；报告导出到 `../results/v2-stage4`。完整校准自动复用相同指纹且哈希正确的折；置换自动复用完成项。分析已完成时拒绝覆盖，应保留原结果并在新工作副本复跑。正在运行时不要启动同一输出目录的第二个进程；只有确认旧进程已停止，才归档部分目录或置换锁文件。完整性门槛为252组合、999次置换、3组UMAP、108个解释模型以及独立审计通过。

冻结配置的准备数据ID也包含环境与源码，其他机器重建数据时可能不同。此时不能跳过核验：先核对72份EDF身份和处理配置一致，再为新运行保存独立配置。以下代码比对已发布记录清单并登记新指纹，保留原 `configs/stage4.json`：

```powershell
& .\.venv\Scripts\python.exe -c "from pathlib import Path; import json; m=json.loads(Path('outputs/prepared-stage3/manifest.json').read_text(encoding='utf-8-sig')); r=json.loads(Path('../results/v2-stage3/manifest.json').read_text(encoding='utf-8-sig')); assert m['fingerprint']['records']==r['fingerprint']['records']; assert m['fingerprint']['config']==r['fingerprint']['config']; c=json.loads(Path('configs/stage4.json').read_text(encoding='utf-8')); c['prepared_stage3_dataset_id']=m['dataset_id']; Path('outputs/stage4-reproduction.json').write_text(json.dumps(c,indent=2),encoding='utf-8')"
```

之后在上面 `runner run`、`runner audit`、`permutation`、`analysis` 和 `scripts/audit_stage4.py` 命令末尾都加 `--config-path outputs/stage4-reproduction.json`。`report_stage4.py`从实际运行记录导出配置。不同平台的逐位一致性不保证，不能将重新运行的新指纹冒充发布时指纹。

第四阶段S3基线与适配方法只在共同24—60秒测试后缀上比较。5/10/20秒是每记录名义上限，完整窗实际覆盖4/10/20秒，合并两记录后实际覆盖8/20/40秒。适配器只接收前缀特征；来源内层验证模拟相同预算，全部目标标签只用于评分。反向CORAL采用目标白化/来源着色，不能宣称与原论文另一方向的正则LR数值等价。

固定C=1的999次置换不是嵌套选参LR的显著性检验。UMAP和脑区解释只作描述，不用于重新选特征、种子或模型。所有结果与局限见[冻结协议](../docs/v2-stage4-protocol.md)。

## 第三阶段复现

先按下方“安装”创建环境并安装本项目。以下完整命令在 `python-v2` 目录执行，适用于尚不存在这些输出目录的工作副本；前3条数据命令建立审计所需的第一阶段准备数据和完整参考预测，已有对应产物时跳过这3条。请将数据路径改为自己的解压目录。不要覆盖本机已有结果，复跑整套实验可使用另一个工作副本。第三阶段新增依赖锁定在 `requirements-stage3-lock.txt`，本机验证的是Windows CPU版PyTorch 2.14.1+cpu；其他设备的运行时间和逐位数值一致性未保证。

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-stage3-lock.txt
& .\.venv\Scripts\python.exe -m pytest -q --junitxml=outputs/stage3-tests.xml
$eegDataDir = '..\..\..\eeg-during-mental-arithmetic-tasks-1.0.0\eeg-during-mental-arithmetic-tasks-1.0.0'
& .\.venv\Scripts\python.exe -m eeg_v2.cli prepare --data-dir $eegDataDir --config configs/stage1.json --output outputs/prepared-stage1
& .\.venv\Scripts\python.exe -m eeg_v2.cli run --prepared outputs/prepared-stage1 --output outputs/loso-stage1
& .\.venv\Scripts\python.exe -m eeg_v2.cli verify --output outputs/loso-stage1
& .\.venv\Scripts\python.exe -m eeg_v2.stage3 prepare --data-dir $eegDataDir --reference-prepared outputs/prepared-stage1 --output outputs/prepared-stage3
& .\.venv\Scripts\python.exe -m eeg_v2.stage3 bridge --prepared outputs/prepared-stage3 --output outputs/bridge-stage3
& .\.venv\Scripts\python.exe scripts/audit_stage3.py --bridge-only --output outputs/stage3-bridge-audit.json
& .\.venv\Scripts\python.exe -m eeg_v2.stage3 run --prepared outputs/prepared-stage3 --output outputs/runs-stage3 --subjects Subject00 Subject17 Subject35 --seeds 42 --workers 3
& .\.venv\Scripts\python.exe -m eeg_v2.stage3 verify --prepared outputs/prepared-stage3 --output outputs/runs-stage3
Copy-Item -LiteralPath outputs/runs-stage3/verification.json -Destination outputs/stage3-smoke-verification.json
& .\.venv\Scripts\python.exe -m eeg_v2.stage3 run --prepared outputs/prepared-stage3 --output outputs/runs-stage3 --workers 4 --resume
& .\.venv\Scripts\python.exe scripts/audit_stage3.py
& .\.venv\Scripts\python.exe scripts/report_stage3.py
```

`--workers`可设1—4；只是同时运行的独立任务数，不改变实验候选。每任务使用2个PyTorch线程。CPU全量预计数小时，准确时间取决于设备和早停轮数。

恢复仅复用配置、代码、环境和数据指纹一致、文件哈希完整的折。中断时没有 `completed.json` 的目录必须先单独保留归档，再从头训练该折；不删除证据、不把部分训练日志当作完整模型。不要同时启动两次针对同一输出目录的训练。

本机现有目录使用 `outputs/prepared-stage3`、`outputs/bridge-stage3`、`outputs/runs-stage3`。一次性收尾流程为 `scripts/pipeline_stage3.py`：先续跑，再全量独立审计，最后导出到 `results/v2-stage3` 并生成报告。只有已有对应准备数据和运行目录且没有其他训练进程时才启动。状态保存在 `outputs/stage3-pipeline-status.json`，训练进度在 `outputs/runs-stage3/summary.json`；最终状态须为 `complete` 才能称全阶段完成。

`scripts/report_stage3.py --bridge-only`可生成明确标注未完成的进度报告；不带此参数时要求108折全部完成并已通过审计。报告脚本使用上述标准目录。模型和输入数组只保留本地，导出不包含EDF、MAT、NPY或模型权重。

## 第二阶段复现

先完成下方环境安装和第一阶段 `prepare`。复现目录名可自行更换，`--reference-prepared` 指向第一阶段已校验的数据产物。以下仍在 `python-v2` 目录执行：

```powershell
$eegDataDir = '..\..\..\eeg-during-mental-arithmetic-tasks-1.0.0\eeg-during-mental-arithmetic-tasks-1.0.0'
& .\.venv\Scripts\python.exe -m eeg_v2.stage2 prepare --data-dir $eegDataDir --reference-prepared outputs/prepared-stage1 --output outputs/prepared-stage2-new
& .\.venv\Scripts\python.exe -m eeg_v2.stage2 run --prepared outputs/prepared-stage2-new --output outputs/smoke-stage2-new --subjects Subject00 Subject17 Subject35
& .\.venv\Scripts\python.exe -m eeg_v2.stage2 run --prepared outputs/prepared-stage2-new --output outputs/runs-stage2-new
& .\.venv\Scripts\python.exe -m eeg_v2.stage2_report --prepared outputs/prepared-stage2-new --runs outputs/runs-stage2-new --output outputs/analysis-stage2-new --stage1-run outputs/loso-stage1
```

第二阶段分析需要第一阶段完整预测，用于检查参考模型概率是否保持一致。如第一阶段使用了其他目录名，修改两个对应参数。只复现一个实验时，可在 `stage2 run` 后加 `--experiments s2_de_svm --subjects Subject00`；单组复核使用 `stage2 verify --prepared ... --output .../s2_de_svm`。最终报告命令要求 18 组实验各完成 36 人，不会把子集当作全量结果。

冻结方案见 [stage2.json](configs/stage2.json)。14 组 S2 和 4 组 S1 是并列实验，每组保持自己明确的训练/验证/测试规则；不是对测试结果自动择优的搜索器。S1 最终模型只拟合训练块，验证块只选参数。S2 的重叠实验只增加源训练窗，验证/测试窗不变。SVM 输出 `decision_margin`，不能解释为概率。

新增 `stage2_features.py`、`stage2_models.py`、`stage2.py`、`stage2_analysis.py`、`stage2_report.py`；本地完整产物分别位于 `outputs/prepared-stage2`、`outputs/runs-stage2`、`outputs/analysis-stage2`。报告证据导出不包含模型和原始 EEG。

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

## 第一阶段冻结协议

配置文件是 [configs/stage1.json](configs/stage1.json)。

- 任务：静息=0、心算=1，36 名被试，19 导联，500 Hz，µV。
- 裁切：每记录前 60 秒。所有 EDF 注释无文字任务事件，实际任务起点无法确认；这是一条固定对接规则。
- 处理：当前窗口内平均参考和去均值；不新增时间滤波或 ICA。供应方已清洗的数据边界不能撤销或重新认证。
- 窗口：2 秒、无重叠；DE 使用 θ 4—8、α 8—13、β 13—30 Hz，频段端点闭合，输出 57 维，按通道优先、频段其次排列。
- S2：外层按人留一，源侧 3 折 GroupKFold；C 候选为 0.01、0.1、1、10，按验证被试 BA 等权均值选优，平局取小 C。标准化每次仅拟合训练数据（ddof=0）。
- 模型决策阈值固定 0.5。测试标签不参与选择、标准化或拟合。
- S1：先按原始时间划块，再切窗口，留 4 秒间隔；第一阶段提供时间清单与隔离测试，第二阶段已完成 S1 模型对照。新增时间滤波后必须重新计算间隔与裁边。
- 失败：记录异常与失败状态，不能用 Dummy 预测掩盖失败；未完成全部请求折时不输出完整平均成绩。

V1 与 V2 的预处理、频段、重叠和选参同时不同。当前数值不能直接解释为某一改动导致的性能增益。EEGNet已在第三阶段完成，有限校准与反向CORAL变体已在第四阶段完成，DANN未实施。

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
