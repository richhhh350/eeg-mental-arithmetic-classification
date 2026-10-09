# 基于 EEG 的静息与心算状态分类

[English](README.md)

**Python V2推荐完整版主线已完成（2026-10-09）。**最后一阶段补齐有限无标签校准、明确方向的反向CORAL变体、来源拟合UMAP、999次整被试标签置换、特征统计与EEGNet脑区扰动，162项测试全部通过。共同测试后缀的主预算结果为：**不校准62.58%、对角标准化67.13%、反向CORAL60.11% BA**，负迁移与统计局限完整保留。入口：[总报告](docs/v2-final-report.md)｜[第四阶段实测](docs/v2-stage4-report.md)｜[复现说明](python-v2/README.md)｜[申请材料事实包](docs/v2-application-brief.zh-CN.md)。DANN、额外ICA和外部验证是未实施的选做扩展。

**Python V2 第三阶段（2026-10-09）：**五组降采样桥接及36人×3个固定种子的EEGNet全部完成，288个模型通过独立审计和模型重放，148项测试通过。EEGNet的被试等权平均平衡准确率为**66.84%**，macro-F1为0.6231，AUC为0.7631；BA均值高于128 Hz的LR/LDA/SVM，但配对描述区间均包含零，不能据此宣称确认性提升。详见[完整报告](docs/v2-stage3-report.md)、[冻结协议](docs/v2-stage3-protocol.md)与[复现命令](python-v2/README.md#第三阶段复现)。

**Python V2 第二阶段（2026-10-08）：**已完成18组实验×36人，包括时间隔离对照、传统模型比较、特征/频段/窗口消融和来源训练PCA。详见[第二阶段报告](docs/v2-stage2-report.md)与[复现命令](python-v2/README.md#第二阶段复现)。比较均按探索性结果解释，原2秒LR参考保持不变。

**Python V2 第一阶段（2026-10-07）：**已实现独立的 [Python 流程](python-v2/README.md)，完成 EDF 溯源、互斥时间划分、数值核验及 36 折嵌套 LOSO 逻辑回归基线。详见[阶段报告](docs/v2-stage1-report.md)。下方 MATLAB V1 与历史结果继续保留。

本课程项目使用公开 EEG 数据，建立预处理、分窗、特征提取和机器学习分类流程，并比较被试内评估与留一被试（LOSO）评估。

## 核心观察

同一被试内的平均测试准确率约为 **98.50%—99.42%**，而在未见过的被试上，平均准确率降至 **52.68%—56.24%**。这一差距提示评估设计与个体差异值得进一步研究；当前结果尚不能支持可靠的通用认知负荷识别器。

| 模型 | 被试内准确率 | LOSO 准确率 | LOSO 平均 F1 |
|---|---:|---:|---:|
| 逻辑回归 | 99.42% | 54.78% | 0.4606 |
| MLP | 98.96% | 56.24% | 0.4824 |
| RBF SVM | 98.50% | 52.68% | 0.3750 |

以上为已保存 CSV 中 36 名被试/36 折指标的算术平均。F1 以心算为正类后按被试平均，不是类别宏平均。发布时未重新运行 MATLAB，也未逐一核验历史 MAT 文件与当前代码配置的对应关系。

## 方法

- 36 名被试、静息与心算两种条件；代码使用 19 导联、500 Hz，并截取每个条件前 60 秒。
- MATLAB + EEGLAB：陷波、带通、平均重参考、ICA 和 ICLabel，提供人工追加剔除入口。
- 1、2、4 秒窗口，分别使用无重叠和 50% 重叠，共六种配置。
- 实现八类特征；当前分类采用 **2 秒窗、50% 重叠、76 维微分熵特征**。
- 被试内按每个条件的时间顺序划分 60%/20%/20% 训练、验证、测试；LOSO 每折用 35 人训练、1 人测试。
- 标准化参数只由训练集计算。被试内使用验证集选择 MLP/SVM 设置，LOSO 使用固定设置。

## 使用方法

环境依赖、数据目录布局和运行顺序见 [英文 README](README.md#running-the-project)。发布版使用相对于脚本的项目路径，未更改模型和信号处理算法。数据来自 [PhysioNet EEGMAT 1.0.0](https://physionet.org/content/eegmat/1.0.0/)，原始 EDF 和中间 MAT 文件不随仓库分发。

```matlab
a2_preprocessing_main
b1_segmentation_feature_extraction
b2_visualization_statistics
c1_subject_dependent_classification
c2_loso_classification
```

预处理包含人工复核弹窗，不是无人值守流程；已存在的预处理文件会跳过。原始运行环境的确切版本未记录，EEGLAB 插件路径可能需要适配。

## 结果边界

静息与心算分类不能直接等同于精确测量认知负荷。窗口在划分前生成，相邻划分边界会共享部分采样点，且滤波和 ICA 在划分前完成。因此，被试内高准确率不能当成跨会话或实际部署表现。

探索性统计将通道和窗口展开后检验，没有充分处理重复测量和相关性；极小 p 值不能证明稳定生物标志物。跨被试结果接近均衡二分类基线，尚未检验是否显著优于机会水平。更多问题与后续方向见 [方法说明](docs/methodological-notes.md)。

## 数据来源

EEG During Mental Arithmetic Tasks，数据 DOI：[10.13026/C2JQ1P](https://doi.org/10.13026/C2JQ1P)。数据由原研究团队采集，并非本项目自行采集。原始研究为 Zyma 等（2019），[Electroencephalograms during Mental Arithmetic Task Performance](https://doi.org/10.3390/data4010014)。使用数据请遵循来源页的许可与引用要求。

本仓库尚未额外授予软件开源许可证；第三方工具与数据各自的许可保持适用。
