"""999 participant-block label exchanges for a PRE-FIXED C=1 S2 baseline."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from eeg_v2.provenance import read_json,write_json,write_csv
from .calibration import fit_lr,ba
from .runner import load_data,fingerprint,initialize_output

_CACHE=None


def init_worker(prepared, config):
    global _CACHE
    with np.load(Path(prepared)/"features.npz",allow_pickle=False) as z:
        x,y,g=z[config["permutation"]["feature"]],z["y"],z["groups"]
    subjects=sorted(set(g)); codes=np.array([subjects.index(s) for s in g])
    folds=[]
    with threadpool_limits(limits=1):
        for i,s in enumerate(subjects):
            train,test=g!=s,g==s
            scaler=StandardScaler().fit(x[train])
            folds.append((scaler.transform(x[train]),scaler.transform(x[test]),y[train],y[test],codes[train],i))
    _CACHE=(folds,config)


def permutation_flips(index, subjects, seed):
    return np.zeros(subjects,dtype=int) if index==0 else np.random.default_rng(np.random.SeedSequence([seed,index])).integers(0,2,subjects)


def one(index):
    folds,config=_CACHE; spec=config["permutation"]
    flips=permutation_flips(index,len(folds),spec["seed"])
    tick=time.perf_counter(); scores=[]
    with threadpool_limits(limits=1):
        for train,test,yt,yv,codes,held in folds:
            model=fit_lr(train,np.bitwise_xor(yt,flips[codes]),spec["C"])
            probability=model.predict_proba(test)[:,1]
            scores.append(ba(np.bitwise_xor(yv,flips[held]),probability))
    return {"index":index,"flips":flips.tolist(),"subject_ba":scores,"mean_subject_ba":float(np.mean(scores)),"elapsed_seconds":time.perf_counter()-tick}


def run(prepared="outputs/prepared-stage3",output="outputs/permutation-stage4",config_path="configs/stage4.json",workers=4,limit=None):
    *_,config,manifest=load_data(prepared,config_path)
    fp=fingerprint(config,manifest,("calibration.py","runner.py","permutation.py"))
    output,fid=initialize_output(output,fp)
    total=config["permutation"]["repetitions"]+1
    request=total if limit is None else min(limit,total)
    lock=output/"run.lock"
    with lock.open("x") as f: f.write(str(os.getpid()))
    try:
        done={}
        for path in output.glob("permutation_*.json"):
            r=read_json(path)
            assert r["fingerprint_id"]==fid
            assert r["flips"]==permutation_flips(r["index"],36,config["permutation"]["seed"]).tolist()
            assert len(r["subject_ba"])==36 and np.isclose(np.mean(r["subject_ba"]),r["mean_subject_ba"])
            done[r["index"]]=r
        tasks=[i for i in range(request) if i not in done]
        with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(prepared,config)) as pool:
            pending={pool.submit(one,i):i for i in tasks}
            for future in as_completed(pending):
                result=future.result(); result["fingerprint_id"]=fid
                path=output/f"permutation_{result['index']:04d}.json"
                tmp=path.with_suffix(".tmp"); write_json(tmp,result); tmp.replace(path)
                done[result["index"]]=result
                write_json(output/"progress.json",{"completed":len(done),"requested":total,"pid":os.getpid(),"status":"running"})
                if len(done)%25==0 or len(done)==request: print(f"Permutation {len(done)}/{total}",flush=True)
        complete=set(done)==set(range(total))
        summary={"status":"complete" if complete else "incomplete","completed":len(done),"requested":total,"fixed_C":config["permutation"]["C"],"fingerprint_id":fid}
        if complete:
            observed=done[0]["mean_subject_ba"]
            exceed=sum(done[i]["mean_subject_ba"]>=observed-1e-12 for i in range(1,total))
            summary.update(observed_ba=observed,permutations=total-1,exceedances=exceed,p_value=(1+exceed)/total,
                           fits=total*36,null_mean=float(np.mean([done[i]["mean_subject_ba"] for i in range(1,total)])),
                           null_sd=float(np.std([done[i]["mean_subject_ba"] for i in range(1,total)],ddof=1)),
                           interpretation="Fixed-C S2 pipeline only; participant condition-exchangeability assumption, not a test of the selected-C pipeline or removal of order confounding")
        write_csv(output/"distribution.csv",[{"index":i,"mean_subject_ba":r["mean_subject_ba"],"elapsed_seconds":r["elapsed_seconds"],"flips":"".join(map(str,r["flips"]))} for i,r in sorted(done.items())])
        write_csv(output/"subject_scores.csv",[{"index":i,"subject":f"Subject{j:02d}","balanced_accuracy":score} for i,r in sorted(done.items()) for j,score in enumerate(r["subject_ba"])])
        write_json(output/"summary.json",summary)
        write_json(output/"progress.json",{"completed":len(done),"requested":total,"status":summary["status"]})
        print({"status":summary["status"],"completed":len(done),"median_seconds_per_permutation":float(np.median([r["elapsed_seconds"] for r in done.values()]))},flush=True)
    finally:
        lock.unlink()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared",default="outputs/prepared-stage3");p.add_argument("--output",default="outputs/permutation-stage4")
    p.add_argument("--config-path",default="configs/stage4.json");p.add_argument("--workers",type=int,default=4);p.add_argument("--limit",type=int)
    run(**vars(p.parse_args()))
