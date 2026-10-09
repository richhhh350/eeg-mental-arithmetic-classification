"""Independent final-stage evidence checks; no new protocol/model selection."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
from scipy import stats
from eeg_v2.provenance import read_json,read_csv,write_json,file_digest,digest
from eeg_v2.stage4.runner import load_data,fingerprint
from eeg_v2.stage4.permutation import init_worker,one,permutation_flips
from eeg_v2.stage4.analysis import adjust_p,sign_flip_p


def main(config_path=None):
    root=Path(__file__).resolve().parents[1]
    x,y,g,_,_,_,config,manifest=load_data(root/"outputs/prepared-stage3",config_path or root/"configs/stage4.json")
    runs=root/"outputs/runs-stage4";perms=root/"outputs/permutation-stage4";analysis=root/"outputs/analysis-stage4"
    runinfo=read_json(runs/"run.json")
    assert runinfo["fingerprint"]==fingerprint(config,manifest)
    assert read_json(runs/"audit.json")["passed"] and read_json(runs/"audit.json")["fingerprint_id"]==runinfo["fingerprint_id"]
    for name in ("predictions.csv","metrics.csv"):
        combined=[]
        for folder in sorted(p for p in runs.glob("Subject*") if p.is_dir()):combined+=read_csv(folder/name)
        actual=read_csv(runs/("subject_metrics.csv" if name=="metrics.csv" else name))
        assert combined==actual
    pinfo=read_json(perms/"run.json");assert pinfo["fingerprint"]==fingerprint(config,manifest,("calibration.py","runner.py","permutation.py"))
    records=[read_json(perms/f"permutation_{i:04d}.json") for i in range(1000)]
    for i,r in enumerate(records):
        assert r["index"]==i and r["fingerprint_id"]==pinfo["fingerprint_id"]
        assert r["flips"]==permutation_flips(i,36,config["permutation"]["seed"]).tolist()
        assert len(r["subject_ba"])==36 and np.allclose(np.mean(r["subject_ba"]),r["mean_subject_ba"],rtol=0,atol=1e-12)
        assert all(0<=s<=1 for s in r["subject_ba"])
    summary=read_json(perms/"summary.json");assert summary["status"]=="complete" and summary["fits"]==36000
    count=sum(r["mean_subject_ba"]>=records[0]["mean_subject_ba"]-1e-12 for r in records[1:])
    assert count==summary["exceedances"] and (count+1)/1000==summary["p_value"]
    assert summary["observed_ba"]==records[0]["mean_subject_ba"]
    distribution=read_csv(perms/"distribution.csv");scores=read_csv(perms/"subject_scores.csv")
    assert len(distribution)==1000 and len(scores)==36000
    for i,r in enumerate(records):
        assert int(distribution[i]["index"])==i and float(distribution[i]["mean_subject_ba"])==r["mean_subject_ba"]
        np.testing.assert_array_equal([float(row["balanced_accuracy"]) for row in scores[i*36:(i+1)*36]],r["subject_ba"])
    # Refit a fixed diagnostic subset; all other completed permutations are retained.
    init_worker(root/"outputs/prepared-stage3",config)
    for index in (0,1,42,999):np.testing.assert_array_equal(one(index)["subject_ba"],records[index]["subject_ba"])
    complete=read_json(analysis/"complete.json")
    assert complete["fingerprint_id"]==digest(fingerprint(config,manifest,("calibration.py","runner.py","analysis.py")))
    for name,sha in complete["files"].items():assert file_digest(analysis/name)==sha
    metrics=read_csv(runs/"subject_metrics.csv");family=read_json(analysis/"primary_comparisons.json")["family"]
    uncorrected=[]
    for item,(a,b) in zip(family,config["primary_pair_family"]):
        aa={r["subject"]:float(r["balanced_accuracy"]) for r in metrics if r["experiment"]==a}
        bb={r["subject"]:float(r["balanced_accuracy"]) for r in metrics if r["experiment"]==b}
        delta=np.array([aa[s]-bb[s] for s in sorted(aa)])
        p=sign_flip_p(delta);uncorrected.append(p)
        assert p==item["sign_flip_p"] and np.isclose(delta.mean()*100,item["mean_difference_pp"],atol=1e-12)
    np.testing.assert_array_equal(adjust_p(uncorrected,"holm"),[r["holm_p"] for r in family])
    d=np.array([x[(g==s)&(y==1)].mean(0)-x[(g==s)&(y==0)].mean(0) for s in sorted(set(g))])
    feature_rows=read_csv(analysis/"feature_condition_statistics.csv");assert len(feature_rows)==57
    np.testing.assert_allclose(d.mean(0),[float(r["mean_de_difference"]) for r in feature_rows],rtol=0,atol=1e-12)
    np.testing.assert_allclose(adjust_p(stats.ttest_1samp(d,0,axis=0).pvalue,"bh"),[float(r["q_bh57"]) for r in feature_rows],rtol=0,atol=1e-12)
    for target in config["umap"]["targets"]:
        model=joblib.load(analysis/f"umap_{target}.joblib");source=g!=target
        np.testing.assert_allclose(model["scaler"].mean_,x[source].mean(0),rtol=0,atol=1e-12)
        np.testing.assert_array_equal(model["umap"]._raw_data,model["scaler"].transform(x[source]).astype(np.float32))
        rows=read_csv(analysis/f"umap_{target}_coordinates.csv");assert len(rows)==2160
        assert sum(r["role"]=="target" for r in rows)==60
    perturb=read_csv(analysis/"eegnet_region_perturbations.csv");means=read_csv(analysis/"eegnet_region_subject_means.csv")
    assert len(perturb)==540 and len(means)==180
    for r in perturb:assert np.isclose(float(r["ba_loss_pp"]),100*(float(r["baseline_ba"])-float(r["perturbed_ba"])),atol=1e-12)
    for r in means:
        values=[float(p["ba_loss_pp"]) for p in perturb if p["region"]==r["region"] and p["subject"]==r["subject"]]
        assert len(values)==3 and np.isclose(np.mean(values),float(r["mean_seed_ba_loss_pp"]),atol=1e-12)
    ea=read_json(analysis/"eegnet_explanation_audit.json");assert ea["passed"] and ea["max_baseline_replay_error"]<=1e-7
    result={"passed":True,"calibration_models":252,"calibration_predictions":9072,"permutation_replicates":999,"permutation_total_fits":36000,
            "permutation_replay_indices":[0,1,42,999],"permutation_p_recomputed":True,"umap_source_fits":3,"paired_statistics_verified":True,
            "feature_tests":57,"eegnet_models_explained":108,"region_perturbations":540,"calibration_fingerprint_id":runinfo["fingerprint_id"],
            "permutation_fingerprint_id":pinfo["fingerprint_id"],"analysis_fingerprint_id":complete["fingerprint_id"]}
    write_json(root/"outputs/stage4-independent-audit.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path")
    main(**vars(parser.parse_args()))
