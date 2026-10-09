"""Export lightweight final-stage evidence and generate the stage report."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eeg_v2.provenance import read_json,read_csv,write_json,write_csv


def main():
    root=Path(__file__).resolve().parents[1];repo=root.parent
    runs=root/"outputs/runs-stage4";analysis=root/"outputs/analysis-stage4";perms=root/"outputs/permutation-stage4"
    audit=read_json(root/"outputs/stage4-independent-audit.json")
    assert audit["passed"] and audit["calibration_models"]==252 and audit["permutation_replicates"]==999
    summary=read_json(runs/"summary.json");null=read_json(perms/"summary.json")
    assert summary["status"]==null["status"]=="complete" and read_json(analysis/"complete.json")["passed"]
    tree=ET.parse(root/"outputs/stage4-tests.xml");suite=tree.find("testsuite")
    assert int(suite.attrib["tests"])==162 and suite.attrib["failures"]==suite.attrib["errors"]=="0"
    suite.attrib.pop("hostname",None)
    dest=repo/"results/v2-stage4";dest.mkdir(exist_ok=True)
    tree.write(dest/"tests.xml",encoding="utf-8",xml_declaration=True)
    write_json(dest/"config.json",read_json(runs/"run.json")["fingerprint"]["config"])
    shutil.copy2(root/"outputs/stage4-independent-audit.json",dest/"independent_audit.json")
    for folder,names in ((runs,["run.json","summary.json","audit.json","subject_metrics.csv","predictions.csv"]),(perms,["run.json","summary.json","distribution.csv","subject_scores.csv"])):
        sub=dest/("calibration" if folder==runs else "permutation");sub.mkdir(exist_ok=True)
        for name in names:shutil.copy2(folder/name,sub/name)
    sub=dest/"analysis";sub.mkdir(exist_ok=True)
    for p in analysis.iterdir():
        if p.is_file() and p.suffix in (".json",".csv",".png") and p.name!="complete.json":shutil.copy2(p,sub/p.name)
    candidates,validation=[],[]
    for folder in sorted(p for p in runs.glob("Subject*") if p.is_dir()):
        selection=read_json(folder/"selection.json")
        for experiment,records in selection["candidates"].items():
            for r in records:
                candidates.append({"target":folder.name,"experiment":experiment,"C":r["C"],"selected":r["C"]==selection["selected_C"][experiment],"mean_validation_subject_ba":r["mean_subject_balanced_accuracy"]})
                validation.extend({"target":folder.name,"experiment":experiment,"C":r["C"],**s} for s in r["scores"])
    assert len(candidates)==1008 and len(validation)==35280
    write_csv(dest/"calibration/selection_summary.csv",candidates);write_csv(dest/"calibration/inner_validation_scores.csv",validation)
    distribution=read_csv(perms/"distribution.csv");values=[float(r["mean_subject_ba"])*100 for r in distribution if int(r["index"])>0]
    fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    ax.hist(values,bins=30,color="#327D99",alpha=.85)
    ax.axvline(null["observed_ba"]*100,color="#C65032",lw=2,label=f"Observed fixed-C LR: {null['observed_ba']*100:.2f}%")
    ax.set(xlabel="Mean participant BA (%)",ylabel="Permutation count",title="999 participant-block label exchanges / fixed C=1 S2 LR")
    ax.legend();fig.savefig(dest/"permutation/permutation_distribution.png",dpi=160);plt.close(fig)
    family=read_json(analysis/"primary_comparisons.json")["family"]
    measured={r["experiment"]:r for r in summary["experiments"]}
    s,n,c=[measured[k] for k in ("standardize_10","none_0","coral_10")]
    paired=read_json(analysis/"paired_descriptive.json")
    features=read_csv(analysis/"feature_condition_statistics.csv");regions=read_csv(analysis/"eegnet_region_subject_means.csv")
    metrics=read_csv(runs/"subject_metrics.csv")
    labels={"none_0":"不校准",**{f"standardize_{b}":f"对角标准化 / {b}秒" for b in (5,10,20)},**{f"coral_{b}":f"反向CORAL / {b}秒" for b in (5,10,20)}}
    lines=["# Python V2 第四阶段：有限校准与完整分析","","日期：2026-10-09。推荐完整版主线已完成；DANN、额外ICA和外部数据验证仍是明确未实施的扩展。见[事前冻结协议](v2-stage4-protocol.md)和[V2总报告](v2-final-report.md)。","",
           "## 实验范围与信息隔离","",
           "7配置×36人全部完成，252个最终模型/适配组合、9072条预测通过独立审计，最大重放误差为0。复用逐窗独立处理的128 Hz、2秒、57维DE；来源内部3折模拟目标校准并选LR的C，最后拟合35名来源者。适配器不接收目标标签或测试后缀。","",
           "| 每记录名义前缀预算 | 两记录实际使用秒数 | 合计校准窗 | 固定测试后缀 | 合计测试窗 |","|---:|---:|---:|---|---:|","| 0秒 | 0 | 0 | 每记录24—60秒 | 36 |","| 5秒 | 8 | 4 | 每记录24—60秒 | 36 |","| 10秒（主预算） | 20 | 10 | 每记录24—60秒 | 36 |","| 20秒 | 40 | 20 | 每记录24—60秒 | 36 |","",
           "5秒预算只使用完整2秒窗的4秒；两个记录边界及两条件均有前缀是划分先验。这是离线分块校准，不能推广为任意自然数据流的无监督适配。所有S3对照共享后缀，与S2整段60秒的分数分表解释。","",
           "CORAL采用事前登记的目标白化/来源着色方向，来源分类器保持固定坐标；对角标准化使用相同中心化和收缩规则。此为反向CORAL变体，在L2正则下不等同于原论文源到目标方向。短样本使用Ledoit–Wolf收缩与固定微小ridge，不按外层成绩调整方向、正则或预算。","",
           "## 全部校准结果","","指标按36名被试等权平均；同一被试在不同预算下不是新增样本。","","| 方法 / 每记录名义预算 | BA | macro-F1 | AUC | 比不校准下降人数 | 下降超过5百分点 |","|---|---:|---:|---:|---:|---:|"]
    for r in summary["experiments"]:
        key=r["experiment"];p=paired.get(key+"_minus_none_0",{})
        lines.append(f"| {labels[key]} | {r['mean_subject_balanced_accuracy']*100:.2f}% | {r['mean_subject_macro_f1']:.4f} | {r['mean_subject_roc_auc']:.4f} | {p.get('negative_transfer_count','—')} | {p.get('drop_more_than_5pp_count','—')} |")
    lines += ["","![校准预算](../results/v2-stage4/analysis/calibration_budget.png)","",
              f"主预算的标准化BA为{s['mean_subject_balanced_accuracy']*100:.2f}%，与不校准相差{(s['mean_subject_balanced_accuracy']-n['mean_subject_balanced_accuracy'])*100:+.2f}个百分点；二者AUC分别为{s['mean_subject_roc_auc']:.4f}与{n['mean_subject_roc_auc']:.4f}。CORAL的BA为{c['mean_subject_balanced_accuracy']*100:.2f}%，与不校准相差{(c['mean_subject_balanced_accuracy']-n['mean_subject_balanced_accuracy'])*100:+.2f}个百分点，与标准化相差{(c['mean_subject_balanced_accuracy']-s['mean_subject_balanced_accuracy'])*100:+.2f}个百分点。保留全部预算和被试，不把负迁移隐藏为失败折。","",
              "结果解释限定在本数据、对齐方向与预算，不能推广为所有CORAL变体的结论。可能因素包括短校准估计误差、前后时段漂移和混合两条件协方差，但本实验未将这些解释逐一识别为因果。","",
              "## 主比较与不确定性","","| 主预算配对比较 | BA平均差（百分点） | 条件性95%描述区间 | 原始符号翻转p | Holm p |","|---|---:|---|---:|---:|"]
    for item in family:
        lo,hi=item["paired_description"]["bootstrap_mean_difference_ci95_pp"]
        lines.append(f"| {item['comparison']} | {item['mean_difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {item['sign_flip_p']:.6f} | {item['holm_p']:.6f} |")
    lines += ["",f"5000次被试配对bootstrap与100000次符号翻转，种子42，Holm仅校正事前声明的3个主比较。标准化−不校准的Holm p为{family[1]['holm_p']:.6f}，CORAL−标准化的Holm p为{family[2]['holm_p']:.6f}。不能仅据未校正区间跨不过零而宣称确认性提升；LOSO训练集重叠、符号交换假设和单数据集探索性限制仍然适用。","","![逐人配对](../results/v2-stage4/analysis/paired_participants.png)","",
              "## 固定LR的正式置换检验","",
              f"固定C=1的DE128+LR、S2整段零校准基线BA为{null['observed_ba']*100:.2f}%。每名被试整体交换或不交换两条件标签，999次置换全部完成，每次重新拟合36折，共计36,000次拟合（含原标签36折）。空分布均值为{null['null_mean']*100:.2f}%，不低于观察统计量的置换为{null['exceedances']}次，按加一公式p={null['p_value']:.3f}。","",
              "此p值依赖被试内整条件交换的可交换性，仅对应固定C流程，不对应嵌套选参LR、EEGNet或域适配模型；固定静息先于心算的顺序/记录混杂没有被消除。0.001是本次999次Monte Carlo检验的分辨率下限，不写成p<0.001。","","![置换分布](../results/v2-stage4/permutation/permutation_distribution.png)","",
              "## 分布与解释","",
              "三个预先指定目标的UMAP均在2100个来源窗口拟合，目标60窗只transform，来源标准化和嵌入状态通过核验。条件颜色存在广泛重叠、被试位置异质；这些是投影描述，不能用聚类图证明泛化或神经机制。","",
              *[f"![UMAP {s}](../results/v2-stage4/analysis/umap_{s}.png)" for s in ("Subject00","Subject17","Subject35")],"",
              f"57个通道×频段以每人条件均值差为单位计算配对t区间、dz和BH-FDR，其中{sum(float(r['q_bh57'])<.05 for r in features)}项q<0.05。全部方向与数值公开，不能将其写成已验证生物标志物；顺序、记录和供应方清洗仍可能贡献差异。","","![特征差异](../results/v2-stage4/analysis/feature_effects.png)","",
              "108个EEGNet保存模型全部重放，5个预定义脑区分别在来源标准化后置零，共540次扰动；每人先平均3种子。","","| 遮蔽脑区 | 平均BA损失（百分点） |","|---|---:|"]
    for region in ("frontal","central","temporal","parietal","occipital"):
        value=np.mean([float(r["mean_seed_ba_loss_pp"]) for r in regions if r["region"]==region]);lines.append(f"| {region} | {value:.2f} |")
    lines += ["","![脑区扰动](../results/v2-stage4/analysis/eegnet_region_perturbation.png)","","![空间权重](../results/v2-stage4/analysis/eegnet_spatial_weights.png)","",
              "脑区包含的通道数不同，置零可能产生分布外信号；损失较大不等于该脑区有更强因果作用。空间权重汇总跨全部滤波器和模型的归一化绝对值，未经跨种子滤波器对齐，也不是脑激活图。","",
              "## 复核、成本与完成边界","",
              f"全部162项测试通过。独立审计复算校准预测、候选均值、置换计数、配对校正与特征统计，核对3个UMAP来源拟合、108模型与540扰动汇总；固定索引0/1/42/999重拟合置换结果一致。校准组合最终拟合耗时中位数为{np.median([float(r['fit_seconds']) for r in metrics]):.4f}秒，36窗预测中位数为{np.median([float(r['prediction_seconds']) for r in metrics]):.6f}秒；不含源侧选参和数据准备，实测硬件为CPU，不能当作通用性能承诺。","",
              "最终审计脚本首次在Windows大小写不敏感的路径匹配中误将subject_metrics.csv视为被试目录；已限制为目录并通过复核。实验算法、冻结方案和结果没有改变。","",
              "推荐完整版主线已经完成。DANN需另行建立同骨干λ=0对照、全部种子和源侧适配选择，不为增加算法名称并入本轮；额外ICA和外部数据验证仍作为扩展。原始EDF、MAT、环境、窗口数组和模型留在本地，发布轻量预测、统计、配置、代码与图表。","",
              "[独立审计](../results/v2-stage4/independent_audit.json)｜[测试记录](../results/v2-stage4/tests.xml)｜[结果状态](../results/v2-stage4/status.json)｜[复现说明](../python-v2/README.md#第四阶段复现)",""]
    (repo/"docs/v2-stage4-report.md").write_text("\n".join(lines),encoding="utf-8")
    status={"exported_at":datetime.now(timezone(timedelta(hours=8))).isoformat(),"full_stage_complete":True,"recommended_v2_mainline_complete":True,"calibration_models":252,"predictions":9072,"permutations":999,"umap_fits":3,"eegnet_models_explained":108,"tests":162,"optional_unimplemented":["DANN","additional_ICA","external_validation"]}
    write_json(dest/"status.json",status);print(status,flush=True)


if __name__=="__main__":main()
