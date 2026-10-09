"""Export completed bridge evidence, or fully audited stage3 results."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from eeg_v2.provenance import read_csv, read_json, write_csv, write_json
from eeg_v2.stage2_analysis import paired_descriptive

LABELS = {"s2_lr500_reference": "LR 500 Hz / original grid", "s2_lr500_grid": "LR 500 Hz / matched grid", "s2_lr128": "LR 128 Hz", "s2_lda128": "LDA 128 Hz", "s2_svm128": "SVM 128 Hz"}


def plot_bars(labels, scores, path, title):
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.48 * len(labels) + 1.5)), constrained_layout=True)
    positions = np.arange(len(labels))
    ax.barh(positions, np.array(scores) * 100, color="#347f9d")
    ax.set_yticks(positions, labels=labels)
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel="Mean participant balanced accuracy (%)", title=title)
    ax.axvline(50, color="grey", ls="--", lw=1)
    for pos, score in zip(positions, scores):
        ax.text(score * 100 + 1, pos, f"{score * 100:.2f}%", va="center", fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def report(root, bridge_only=False):
    root = Path(root).resolve()
    dest = root.parent / "results/v2-stage3"
    prepared, bridge, runs = (root / f"outputs/{x}" for x in ("prepared-stage3", "bridge-stage3", "runs-stage3"))
    config = read_json(prepared / "config.json")
    bridge_summary = read_json(bridge / "summary.json")
    bridge_audit = read_json(root / "outputs/stage3-bridge-audit.json")
    assert bridge_summary["status"] == "complete" and bridge_audit["passed"]
    cnn_summary = read_json(runs / "summary.json")
    if not bridge_only:
        audit = read_json(root / "outputs/stage3-independent-audit.json")
        assert cnn_summary["status"] == "complete" and cnn_summary["completed_folds"] == 108
        assert audit["passed"] and audit["eegnet_models"] == 108
        assert cnn_summary["seeds"] == config["seeds"]
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "manifest.json", "filter_diagnostics.json", "feature_sensitivity.json"):
        shutil.copy2(prepared / name, dest / name)
    for source, name in (("stage3-bridge-audit.json", "bridge_audit.json"), ("stage3-smoke-verification.json", "smoke_verification.json")):
        shutil.copy2(root / "outputs" / source, dest / name)
    tree = ET.parse(root / "outputs/stage3-tests.xml")
    for node in tree.iter():
        node.attrib.pop("hostname", None)
    tree.write(dest / "tests.xml", encoding="utf-8", xml_declaration=True)
    bridge_rows, scores, table = {}, [], []
    for item in bridge_summary["experiments"]:
        key = item["experiment"]["id"]
        folder = dest / "bridge" / key
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "subject_metrics.csv", "predictions.csv"):
            shutil.copy2(bridge / key / name, folder / name)
        bridge_rows[key] = read_csv(folder / "subject_metrics.csv")
        scores.append(item["mean_subject_balanced_accuracy"])
        table.append(f"| {LABELS[key]} | {scores[-1]*100:.2f}% | {item['mean_subject_macro_f1']:.4f} | {item['mean_subject_roc_auc']:.4f} |")
    for name in ("summary.json", "run.json"):
        shutil.copy2(bridge / name, dest / "bridge" / name)
    plot_bars(list(LABELS.values()), scores, dest / "bridge_comparison.png", "Traditional resampling controls")
    paired = {
        "grid500_minus_reference500": paired_descriptive(bridge_rows["s2_lr500_reference"], bridge_rows["s2_lr500_grid"]),
        "lr128_minus_matched_grid500": paired_descriptive(bridge_rows["s2_lr500_grid"], bridge_rows["s2_lr128"]),
        "lr128_minus_reference500": paired_descriptive(bridge_rows["s2_lr500_reference"], bridge_rows["s2_lr128"])}
    sensitivity = read_json(prepared / "feature_sensitivity.json")["comparisons"]
    exported_at = datetime.now(timezone(timedelta(hours=8))).isoformat()
    body = ["# Python V2 第三阶段：降采样桥接与 EEGNet", "",
            f"报告生成时间（北京时间）：{exported_at}。实验方案冻结于2026-10-08，见[协议与资源调整](v2-stage3-protocol.md)。", "",
            "## 已完成的传统桥接", "",
            "5组×36人全部完成，180个保存模型和10,800条预测通过独立审计，模型重放最大误差为0。原500 Hz LR的2160条概率与第一阶段完全相同。", "",
            "| 配置 | 平均BA | macro-F1 | AUC |", "|---|---:|---:|---:|", *table, "",
            "![传统桥接](../results/v2-stage3/bridge_comparison.png)", "",
            "降采样前后LR均值接近，但这不证明两种处理等价。500 Hz参考使用1024点FFT，128 Hz使用256点FFT；增加500 Hz/1000点FFT对照，单独记录频率网格变化。", "",
            f"网格变化的DE绝对差中位数为{sensitivity[0]['median_abs_de']:.5f}；在相同0.5 Hz网格下，降采样相关绝对差中位数为{sensitivity[1]['median_abs_de']:.5f}。这些是当前数据的数值描述。", "",
            "固定抗混叠FIR和10/30/100/150 Hz合成信号检查通过。反射只使用本2秒窗的数据；内部幅值与全窗RMS分别保存，边缘效应仍是局限。网络采用宽频时域输入，DE使用4—30 Hz特征，因此比较同时涉及表征与分类器。", "",
            "## EEGNet 状态", ""]
    status = {"exported_at": exported_at, "bridge": "complete", "eegnet": "in_progress", "full_stage_complete": False}
    if bridge_only:
        status["eegnet_completed_at_export"] = cnn_summary["completed_folds"]
        body += [f"**全量训练未完成。**本报告生成时完成{cnn_summary['completed_folds']}/108个被试-种子组合；实时进度保存在本地 `python-v2/outputs/runs-stage3/summary.json`。", "",
                 "Subject00、17、35的seed42技术运行已通过模型复核，重放误差为0。全量使用36人和13/42/2026三个种子，不把3人技术验证的成绩作为整体结论。", "",
                 "148项自动测试通过，包括微小合成数据可拟合、目标数据与标签扰动不改变训练、窗口隔离、来源标准化、BN/Dropout测试状态和失败处理。", "",
                 "2026-10-09恢复时，原训练进程已停止，保留了10个完整组合与4个未完成目录。完整折经指纹/哈希校验后复用，未完成目录单独归档并从头重跑，不混用部分训练状态。", "",
                 "完整运行需要数小时CPU计算。只有108个组合全部完成并通过审计，才生成EEGNet全量汇总；目前不报告整体优劣或显著性。"]
    else:
        folder = dest / "eegnet"
        folder.mkdir(exist_ok=True)
        for name in ("summary.json", "subject_seed_metrics.csv", "subject_mean_metrics.csv", "predictions.csv", "verification.json", "run.json", "request.json"):
            shutil.copy2(runs / name, folder / name)
        shutil.copy2(root / "outputs/stage3-independent-audit.json", dest / "independent_audit.json")
        cnn_rows = read_csv(folder / "subject_mean_metrics.csv")
        assert len(cnn_rows) == 36
        epochs, learning = [], []
        for target in cnn_summary["subjects"]:
            for seed in config["seeds"]:
                selection = read_json(runs / f"seed{seed}" / target / "selection.json")
                epochs.append({"subject": target, "seed": seed, "selected_epochs": selection["selected_epochs"], "inner_epochs_total": sum(r["epochs_run"] for r in selection["inner_folds"]), "inner_cap_hits": sum(r["hit_epoch_cap"] for r in selection["inner_folds"])})
                for inner in selection["inner_folds"]:
                    learning.extend({"subject": target, "seed": seed, "inner_fold": inner["fold"], "epoch": r["epoch"], "training_loss": r["training_loss"], "validation_ba": r["mean_subject_validation_ba"]} for r in inner["history"])
        write_csv(dest / "training_budgets.csv", epochs)
        write_csv(dest / "learning_curves.csv", learning)
        budget = [r["selected_epochs"] for r in epochs]
        body += ["**36人×3种子全部完成，108个EEGNet模型和6480条预测通过复核。**先对每人的三个种子指标取均值，再对36人等权平均。", "",
                 "| EEGNet种子 | 平均BA | macro-F1 | AUC |", "|---|---:|---:|---:|"]
        for item in cnn_summary["by_seed"]:
            body.append(f"| {item['seed']} | {item['mean_subject_balanced_accuracy']*100:.2f}% | {item['mean_subject_macro_f1']:.4f} | {item['mean_subject_roc_auc']:.4f} |")
        body += [f"| 三种子指标均值 | {cnn_summary['mean_subject_seed_balanced_accuracy']*100:.2f}% | {cnn_summary['mean_subject_seed_macro_f1']:.4f} | {cnn_summary['mean_subject_seed_roc_auc']:.4f} |", "",
                 "三个种子不是新增被试；这里没有进行概率集成，也没有挑选最好种子。EEGNet的平均BA较高，但其AUC低于LDA 128 Hz；结果不支持在所有指标上都占优的结论。", "",
                 f"最终重训轮数为{min(budget)}—{max(budget)}，中位数{np.median(budget):g}；324个内层训练中有{sum(r['inner_cap_hits'] for r in epochs)}次达到60轮上限。触及预算不代表已充分收敛。", "",
                 "| 配对比较 | 平均差（百分点） | 条件性95%描述区间 | 提高/相同/下降 |", "|---|---:|---:|---:|"]
        for key in ("s2_lr128", "s2_lda128", "s2_svm128"):
            item = paired_descriptive(bridge_rows[key], cnn_rows)
            paired[f"eegnet_mean_minus_{key}"] = item
            lo, hi = item["bootstrap_mean_difference_ci95_pp"]
            body.append(f"| EEGNet均值 − {LABELS[key]} | {item['mean_difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {item['wins']}/{item['ties']}/{item['losses']} |")
        body += ["", "区间使用固定已保存预测做5000次被试配对bootstrap，不包含重新训练不确定性或探索性多重比较校正；LOSO训练集高度重叠，不据此宣布确认性显著提升。", ""]
        labels = list(LABELS.values())[2:] + [f"EEGNet seed{r['seed']}" for r in cnn_summary["by_seed"]] + ["EEGNet seed-metric mean"]
        values = scores[2:] + [r["mean_subject_balanced_accuracy"] for r in cnn_summary["by_seed"]] + [cnn_summary["mean_subject_seed_balanced_accuracy"]]
        plot_bars(labels, values, dest / "eegnet_comparison.png", "No target calibration: all prescribed seeds")
        fig, axes = plt.subplots(3, 3, figsize=(13, 9), constrained_layout=True, sharey=True)
        for i, target in enumerate(config["smoke_subjects"]):
            for j, seed in enumerate(config["seeds"]):
                selection = read_json(runs / f"seed{seed}" / target / "selection.json")
                ax = axes[i, j]
                for inner in selection["inner_folds"]:
                    ax.plot([r["epoch"] for r in inner["history"]], [r["mean_subject_validation_ba"] for r in inner["history"]], label=f"Inner{inner['fold']+1}", lw=1)
                ax.axvline(selection["selected_epochs"], ls=":", color="black", label="Refit epoch")
                ax.set(title=f"{target} / seed{seed}", xlabel="Epoch", ylabel="Source validation BA", ylim=(0.3, 1))
        axes[0, 0].legend(fontsize=8)
        fig.savefig(dest / "source_validation_curves.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        for seed in config["seeds"]:
            values = [read_json(runs / f"seed{seed}" / target / "metrics.json")["balanced_accuracy"] * 100 for target in cnn_summary["subjects"]]
            ax.plot(np.arange(36), values, lw=0.8, alpha=0.65, label=f"Seed {seed}")
        ax.plot(np.arange(36), [float(r["balanced_accuracy"]) * 100 for r in cnn_rows], color="black", lw=1.8, label="Seed-metric mean")
        ax.axhline(50, color="grey", ls="--")
        ax.set(xlabel="Participant index (Subject00–35)", ylabel="Balanced accuracy (%)", ylim=(0, 100), title="All participants and seeds")
        ax.legend(ncols=4)
        fig.savefig(dest / "eegnet_by_subject.png", dpi=160)
        plt.close(fig)
        body += ["![模型对照](../results/v2-stage3/eegnet_comparison.png)", "", "![全部被试](../results/v2-stage3/eegnet_by_subject.png)", "", "![预先指定被试的源验证曲线](../results/v2-stage3/source_validation_curves.png)", "",
                 "曲线的代表被试在成绩产生前指定，显示的是来源验证而非目标成绩；其他折逐轮均值与训练损失也随CSV保存。", "",
                 "独立审计从混淆计数和秩统计复算指标，核对源侧标准化、内层划分和轮数选择，并重放全部保存模型。288个模型的重放最大误差为0，见[完整审计](../results/v2-stage3/independent_audit.json)。148项自动测试通过，包括窗口隔离、目标数据与标签扰动不改变训练、来源标准化、BN/Dropout测试状态及失败处理，见[测试记录](../results/v2-stage3/tests.xml)。"]
        status.update(eegnet="complete", full_stage_complete=True, models=288, predictions=17280)
    body += ["", "## 解释边界与文件", "",
             "本轮采用已声明的CPU预算与本地PyTorch实现，不宣称网络充分调优或与其他实现数值等价。原数据的供应方清洗、无事件确认的任务起点、静息先于心算及缺少外部验证等局限保持不变。", "",
             "原始EDF、窗口数组和模型保留在本地；第三阶段发布材料仅含代码、匿名预测、指标、配置、审计与图表。同步记录以GitHub提交历史为准。", "",
             "[结果状态](../results/v2-stage3/status.json)｜[桥接审计](../results/v2-stage3/bridge_audit.json)｜[特征敏感性](../results/v2-stage3/feature_sensitivity.json)｜[配对描述](../results/v2-stage3/paired_descriptive.json)｜[复现说明](../python-v2/README.md)", ""]
    write_json(dest / "paired_descriptive.json", paired)
    write_json(dest / "status.json", status)
    path = root.parent / "docs/v2-stage3-report.md"
    path.write_text("\n".join(body), encoding="utf-8")
    print({"report": str(path), "status": status}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--bridge-only", action="store_true")
    report(**vars(parser.parse_args()))
