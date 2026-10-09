"""Predeclared descriptive statistics, source-fitted UMAP and EEGNet perturbations."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from eeg_v2.data import CHANNELS
from eeg_v2.provenance import read_json,read_csv,write_json,write_csv,file_digest
from eeg_v2.stage2_analysis import paired_descriptive
from .runner import load_data,fingerprint,initialize_output


def adjust_p(p,method):
    p=np.asarray(p,dtype=float)
    if p.ndim!=1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError("Invalid p values")
    order=np.argsort(p); ranked=p[order]; n=len(p)
    if method=="holm": adjusted=np.maximum.accumulate(ranked*np.arange(n,0,-1))
    elif method=="bh": adjusted=np.minimum.accumulate((ranked*n/np.arange(1,n+1))[::-1])[::-1]
    else: raise ValueError("Unknown correction")
    result=np.empty(n);result[order]=np.minimum(adjusted,1)
    return result


def sign_flip_p(differences,repetitions=100000,seed=42):
    d=np.asarray(differences,dtype=float)
    rng=np.random.default_rng(seed); threshold=abs(d.mean()); exceed=0
    for start in range(0,repetitions,5000):
        signs=rng.integers(0,2,size=(min(5000,repetitions-start),len(d)),dtype=np.int8)*2-1
        exceed+=int(np.sum(np.abs((signs*d).mean(1))>=threshold-1e-12))
    return (exceed+1)/(repetitions+1)


def paired_results(runs,config,output):
    rows=read_csv(Path(runs)/"subject_metrics.csv")
    paired={}
    baseline=[r for r in rows if r["experiment"]=="none_0"]
    for method in ("standardize","coral"):
        for budget in config["budgets_seconds_per_record"]:
            key=f"{method}_{budget}"; current=[r for r in rows if r["experiment"]==key]
            entry=paired_descriptive(baseline,current)
            by_subject={r["subject"]:float(r["balanced_accuracy"]) for r in baseline}
            d=np.array([float(r["balanced_accuracy"])-by_subject[r["subject"]] for r in current])
            entry.update(negative_transfer_count=int((d < -1e-12).sum()),drop_more_than_5pp_count=int((d < -.05-1e-12).sum()),
                         negative_transfer_fraction=float((d < -1e-12).mean()),test_windows_per_participant=36)
            paired[key+"_minus_none_0"]=entry
    family=[]
    for a,b in config["primary_pair_family"]:
        ra,rb=[r for r in rows if r["experiment"]==a],[r for r in rows if r["experiment"]==b]
        mb={r["subject"]:float(r["balanced_accuracy"]) for r in rb}
        d=np.array([float(r["balanced_accuracy"])-mb[r["subject"]] for r in ra])
        family.append({"comparison":f"{a}_minus_{b}","mean_difference_pp":float(d.mean()*100),"sign_flip_p":sign_flip_p(d),"paired_description":paired_descriptive(rb,ra)})
    for r,p in zip(family,adjust_p([r["sign_flip_p"] for r in family],"holm")):r["holm_p"]=float(p)
    write_json(output/"paired_descriptive.json",paired)
    write_json(output/"primary_comparisons.json",{"family":family,"interpretation":"Approximate paired sign exchange with overlapping LOSO fits; exploratory, not independent replication. Primary family specified before stage4 scores.","repetitions":100000,"seed":42})
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    base=np.mean([float(r["balanced_accuracy"]) for r in baseline])*100
    for ax in axes: ax.set_xticks([5,10,20]);ax.set_xlabel("Nominal calibration seconds per record")
    axes[0].axhline(base,color="gray",linestyle="--",label="No calibration / same suffix")
    for method,color in (("standardize","#2878B5"),("coral","#E07A1F")):
        means=[100*np.mean([float(r["balanced_accuracy"]) for r in rows if r["experiment"]==f"{method}_{b}"]) for b in (5,10,20)]
        negative=[paired[f"{method}_{b}_minus_none_0"]["negative_transfer_count"] for b in (5,10,20)]
        axes[0].plot([5,10,20],means,"o-",color=color,label=method)
        axes[1].plot([5,10,20],negative,"o-",color=color,label=method)
    axes[0].set(ylabel="Mean participant balanced accuracy (%)",ylim=(0,100),title="Common 24-60 s test suffix")
    axes[1].set(ylabel="Participants with lower BA than no calibration",ylim=(0,36),title="Negative transfer (all 36 participants)")
    for ax in axes:ax.legend(fontsize=8)
    fig.savefig(output/"calibration_budget.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    for i,(a,b) in enumerate(config["primary_pair_family"]):
        aa={r["subject"]:float(r["balanced_accuracy"]) for r in rows if r["experiment"]==a}
        bb={r["subject"]:float(r["balanced_accuracy"]) for r in rows if r["experiment"]==b}
        d=np.array([100*(aa[s]-bb[s]) for s in sorted(aa)])
        ax.scatter(np.linspace(i-.16,i+.16,36),d,s=16,alpha=.55)
        ax.plot([i-.2,i+.2],[d.mean(),d.mean()],color="black",lw=2)
    ax.axhline(0,color="gray",ls="--");ax.set_xticks(range(3),[f"{a}\nminus {b}" for a,b in config["primary_pair_family"]])
    ax.set(ylabel="Participant BA difference (percentage points)",title="Primary 10 s/record comparisons; all participants")
    fig.savefig(output/"paired_participants.png",dpi=160);plt.close(fig)


def feature_statistics(x,y,groups,output):
    subjects=sorted(set(groups))
    d=np.array([x[(groups==s)&(y==1)].mean(0)-x[(groups==s)&(y==0)].mean(0) for s in subjects])
    mean,sd=d.mean(0),d.std(0,ddof=1); se=sd/np.sqrt(len(subjects)); critical=stats.t.ppf(.975,len(subjects)-1)
    p=stats.ttest_1samp(d,0,axis=0).pvalue;q=adjust_p(p,"bh")
    rows=[]
    for i in range(57):
        rows.append({"channel":CHANNELS[i//3],"band":["theta","alpha","beta"][i%3],"participants":36,"mean_de_difference":float(mean[i]),"ci95_low":float(mean[i]-critical*se[i]),"ci95_high":float(mean[i]+critical*se[i]),"dz":float(mean[i]/sd[i]),"p_uncorrected":float(p[i]),"q_bh57":float(q[i])})
    write_csv(output/"feature_condition_statistics.csv",rows)
    write_csv(output/"participant_feature_differences.csv",[{"subject":s,"channel":CHANNELS[i//3],"band":["theta","alpha","beta"][i%3],"arithmetic_minus_rest_de":float(d[j,i])} for j,s in enumerate(subjects) for i in range(57)])
    fig,ax=plt.subplots(figsize=(5.5,8),constrained_layout=True);value=(mean/sd).reshape(19,3);v=max(.1,abs(value).max())
    im=ax.imshow(value,cmap="RdBu_r",vmin=-v,vmax=v,aspect="auto")
    ax.set_yticks(range(19),CHANNELS);ax.set_xticks(range(3),["Theta","Alpha","Beta"])
    ax.set_title("Participant-paired DE differences\nArithmetic minus rest; * BH q < .05")
    for i in range(19):
        for j in range(3):
            ax.text(j,i,f"{value[i,j]:.2f}"+("*" if q[i*3+j]<.05 else ""),ha="center",va="center",fontsize=8,color="white" if abs(value[i,j])>v*.6 else "black")
    fig.colorbar(im,ax=ax,label="Paired standardized difference dz")
    fig.savefig(output/"feature_effects.png",dpi=160);plt.close(fig)


def umap_sources(x,y,groups,config,output):
    import umap
    summaries=[]
    spec=config["umap"]
    for target in spec["targets"]:
        source=groups!=target;scaler=StandardScaler().fit(x[source]);z=scaler.transform(x)
        model=umap.UMAP(n_neighbors=spec["n_neighbors"],min_dist=spec["min_dist"],n_components=2,n_epochs=spec["n_epochs"],metric=spec["metric"],random_state=42,transform_seed=42,n_jobs=1)
        with threadpool_limits(limits=1):model.fit(z[source])
        before=model.embedding_.copy();raw=model._raw_data.copy()
        coordinates=np.empty((len(x),2));coordinates[source]=model.embedding_
        coordinates[~source]=model.transform(z[~source])
        assert np.array_equal(model.embedding_,before) and np.array_equal(model._raw_data,raw)
        np.testing.assert_allclose(raw,z[source].astype(np.float32),rtol=0,atol=0)
        joblib.dump({"scaler":scaler,"umap":model},output/f"umap_{target}.joblib")
        write_csv(output/f"umap_{target}_coordinates.csv",[{"subject":str(s),"label":int(label),"role":"source" if train else "target","umap1":float(c[0]),"umap2":float(c[1])} for s,label,train,c in zip(groups,y,source,coordinates)])
        fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
        for label,color,name in ((0,"#2878B5","Rest"),(1,"#E07A1F","Arithmetic")):
            for train in (True,False):
                mask=(y==label)&(source==train)
                axes[0].scatter(coordinates[mask,0],coordinates[mask,1],s=8 if train else 32,alpha=.25 if train else .9,c=color,marker="o" if train else "^",edgecolors="none" if train else "black",linewidths=.4,label=name+(" / sources" if train else " / target"))
        codes=np.array([int(s[-2:]) for s in groups])
        im=axes[1].scatter(coordinates[source,0],coordinates[source,1],c=codes[source],cmap="turbo",vmin=0,vmax=35,s=8,alpha=.5)
        axes[1].scatter(coordinates[~source,0],coordinates[~source,1],c=codes[~source],cmap="turbo",vmin=0,vmax=35,s=32,marker="^",edgecolors="black",linewidths=.5)
        fig.colorbar(im,ax=axes[1],label="Participant index")
        axes[0].legend(fontsize=7);axes[0].set_title(f"Source-fitted UMAP / held out {target}");axes[1].set_title("Participant colors; target triangles")
        for ax in axes:ax.set(xlabel="UMAP 1",ylabel="UMAP 2")
        fig.savefig(output/f"umap_{target}.png",dpi=160);plt.close(fig)
        summaries.append({"target":target,"source_windows":int(source.sum()),"target_windows":int((~source).sum()),"fit_subjects":sorted(set(groups[source])),"target_transform_preserved_source_embedding":True,"raw_source_fit_verified":True,"labels_used_for_fit":False})
        print(f"UMAP {target}: source fit and target transform verified",flush=True)
    write_json(output/"umap_audit.json",{"passed":True,"fits":summaries,"parameters":spec,"interpretation":"Descriptive projection; no model selection, no proof of physiological separation"})


def eegnet_explanations(prepared,config,output,runs="outputs/runs-stage3"):
    import torch
    from eeg_v2.eegnet import EEGNet,ChannelScaler,infer
    torch.set_num_threads(2)
    x=np.load(Path(prepared)/"windows128.npy",allow_pickle=False)
    with np.load(Path(prepared)/"features.npz",allow_pickle=False) as f:y,g=f["y"],f["groups"]
    original=read_json(Path(prepared)/"config.json");runs=Path(runs)
    assert sorted(c for cs in config["regions"].values() for c in cs)==sorted(CHANNELS)
    perturb,weights,errors=[],[],[]
    for seed in original["seeds"]:
        for subject in sorted(set(g)):
            folder=runs/f"seed{seed}"/subject;completed=read_json(folder/"completed.json")
            for name,sha in completed["files"].items():assert file_digest(folder/name)==sha
            model=EEGNet(**original["model"]);model.load_state_dict(torch.load(folder/"model.pt",weights_only=True,map_location="cpu"))
            scaler=ChannelScaler()
            with np.load(folder/"scaler.npz",allow_pickle=False) as z:scaler.mean_,scaler.scale_=z["mean"],z["scale"]
            mask=g==subject;z=scaler.transform(x[mask]);labels=y[mask]
            baseline=infer(model,z);recorded=read_csv(folder/"predictions.csv")
            errors.append(float(np.max(abs(baseline-[float(r["score"]) for r in recorded]))))
            baseline_ba=float(np.mean((baseline>=.5)==labels))
            for region,channels in config["regions"].items():
                changed=z.copy();changed[:,[CHANNELS.index(c) for c in channels],:]=0
                probability=infer(model,changed);score=float(np.mean((probability>=.5)==labels))
                perturb.append({"subject":str(subject),"seed":seed,"region":region,"baseline_ba":baseline_ba,"perturbed_ba":score,"ba_loss_pp":100*(baseline_ba-score),"mean_absolute_probability_change":float(np.mean(abs(baseline-probability)))})
            spatial=model.spatial.weight.detach().numpy()[:,0,:,0]
            magnitude=np.abs(spatial).mean(0);magnitude/=magnitude.sum()
            weights.extend({"subject":str(subject),"seed":seed,"channel":channel,"normalized_absolute_spatial_weight":float(magnitude[i])} for i,channel in enumerate(CHANNELS))
        print(f"EEGNet explanations: all36 subjects / seed{seed}",flush=True)
    assert len(perturb)==540 and max(errors)<=1e-7
    write_csv(output/"eegnet_region_perturbations.csv",perturb);write_csv(output/"eegnet_spatial_weights.csv",weights)
    means=[{"subject":str(s),"region":region,"mean_seed_ba_loss_pp":float(np.mean([r["ba_loss_pp"] for r in perturb if r["subject"]==s and r["region"]==region]))} for s in sorted(set(g)) for region in config["regions"]]
    write_csv(output/"eegnet_region_subject_means.csv",means)
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    for i,region in enumerate(config["regions"]):
        values=[r["mean_seed_ba_loss_pp"] for r in means if r["region"]==region]
        ax.scatter(np.linspace(i-.17,i+.17,36),values,s=17,alpha=.6)
        ax.plot([i-.2,i+.2],[np.mean(values)]*2,c="black",lw=2)
    ax.axhline(0,color="gray",ls="--");ax.set_xticks(range(5),list(config["regions"]))
    ax.set(ylabel="Original BA minus masked BA (percentage points)",title="EEGNet region masking: each participant averages all3 seeds")
    fig.savefig(output/"eegnet_region_perturbation.png",dpi=160);plt.close(fig)
    values=np.array([[r["normalized_absolute_spatial_weight"] for r in weights if r["channel"]==c] for c in CHANNELS])
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    ax.bar(range(19),values.mean(1),color="#327D99");ax.set_xticks(range(19),CHANNELS,rotation=45)
    ax.set(ylabel="Mean within-model normalized absolute weight",title="All108 EEGNet models / all spatial filters; descriptive weights")
    fig.savefig(output/"eegnet_spatial_weights.png",dpi=160);plt.close(fig)
    write_json(output/"eegnet_explanation_audit.json",{"passed":True,"models":108,"perturbations":540,"max_baseline_replay_error":max(errors),"input":"Zero after frozen source scaling; 5 predefined disjoint channel regions","interpretation":"Potentially out-of-distribution perturbation and unaligned absolute weights; not causal brain-region evidence"})


def run(prepared="outputs/prepared-stage3",runs="outputs/runs-stage4",output="outputs/analysis-stage4",config_path="configs/stage4.json"):
    x,y,g,_,_,_,config,manifest=load_data(prepared,config_path)
    assert read_json(Path(runs)/"audit.json")["passed"]
    output,fid=initialize_output(output,fingerprint(config,manifest,("calibration.py","runner.py","analysis.py")))
    if (output/"complete.json").exists():raise FileExistsError("Analysis already complete; keep existing evidence")
    paired_results(runs,config,output);feature_statistics(x,y,g,output)
    umap_sources(x,y,g,config,output);eegnet_explanations(prepared,config,output)
    write_json(output/"complete.json",{"passed":True,"fingerprint_id":fid,"files":{p.name:file_digest(p) for p in output.iterdir() if p.is_file()}})
    print("Stage4 statistics, UMAP and EEGNet explanations complete",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared",default="outputs/prepared-stage3");p.add_argument("--runs",default="outputs/runs-stage4")
    p.add_argument("--output",default="outputs/analysis-stage4");p.add_argument("--config-path",default="configs/stage4.json")
    run(**vars(p.parse_args()))
