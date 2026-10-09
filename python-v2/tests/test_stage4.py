from copy import deepcopy
import numpy as np
import pytest
import joblib
from eeg_v2.provenance import read_json
from eeg_v2.stage4.calibration import fit_adapter, covariance, symmetric_power, window_masks, fit_final, predict
from eeg_v2.stage4 import runner
from eeg_v2.stage4.analysis import adjust_p, sign_flip_p
from eeg_v2.stage4.permutation import permutation_flips


@pytest.mark.parametrize("method", ["standardize", "coral"])
def test_regularized_transport_with_four_samples_57_dimensions(method):
    rng = np.random.default_rng(42)
    source, target = rng.normal(size=(120,57)), rng.normal(size=(4,57))*3+2
    adapter = fit_adapter(source, target, method)
    _, a, _ = covariance(source); _, b, _ = covariance(target)
    if method == "standardize": a, b = np.diag(np.diag(a)), np.diag(np.diag(b))
    np.testing.assert_allclose(adapter.transform_matrix.T@b@adapter.transform_matrix, a, atol=1e-9)
    np.testing.assert_allclose(adapter.transform(target).mean(0), source.mean(0), atol=1e-12)
    assert np.isfinite(adapter.transform(target)).all()


def test_diagonal_transport_matches_analytic_scaling():
    s, t = np.diag([4.,9.]), np.diag([16.,1.])
    np.testing.assert_allclose(symmetric_power(t,-.5)@symmetric_power(s,.5), np.diag([.5,3]))


def test_calibration_row_order_and_label_absence():
    rng = np.random.default_rng(7); x, c = rng.normal(size=(100,8)), rng.normal(size=(10,8))
    a, b = fit_adapter(x,c,"coral"), fit_adapter(x,c[::-1],"coral")
    np.testing.assert_allclose(a.transform(x), b.transform(x), atol=1e-12)
    with pytest.raises(TypeError): fit_adapter(x,c,"coral",target_y=np.arange(10)%2)


def test_no_calibration_forbids_target_input():
    x=np.arange(12).reshape(4,3)
    np.testing.assert_array_equal(fit_adapter(x,None,"none").transform(x), x)
    with pytest.raises(ValueError): fit_adapter(x,x,"none")


@pytest.mark.parametrize("bad", [np.ones((1,4)), np.full((4,4),np.nan)])
def test_invalid_calibration_fails_without_dummy(bad):
    with pytest.raises(ValueError): fit_adapter(np.ones((10,4)),bad,"coral")


def test_five_second_cap_uses_only_complete_windows():
    config=read_json("configs/stage4.json")
    rows=[{"start_sample":i*500,"stop_sample":(i+2)*500,"fs_hz":500} for i in range(0,60,2)]
    cal,test=window_masks(rows,config)
    assert [int(cal[b].sum()) for b in (5,10,20)] == [2,5,10]
    assert test.sum()==18 and np.flatnonzero(test)[0]==12
    assert all(not (m & test).any() for m in cal.values())
    config["test_start_seconds"]=18
    with pytest.raises(ValueError): window_masks(rows,config)


def test_prediction_batch_does_not_refit_adapter():
    rng=np.random.default_rng(4); x=rng.normal(size=(100,5)); y=(x[:,0]>0).astype(int)
    c=rng.normal(size=(4,5)); t=rng.normal(size=(8,5))
    model=fit_final(x,y,c,"coral",1.)
    before=deepcopy(model["adapter"])
    a=predict(model,t)
    b=np.array([predict(model,row[None])[0] for row in t])
    np.testing.assert_allclose(a,b,atol=1e-12)
    np.testing.assert_array_equal(before.transform_matrix,model["adapter"].transform_matrix)


def test_outer_target_suffix_and_labels_cannot_change_fitted_state(tmp_path,monkeypatch):
    config=read_json("configs/stage4.json")
    rng=np.random.default_rng(31)
    groups=np.repeat([f"Subject{i:02d}" for i in range(6)],60)
    y=np.tile(np.repeat([0,1],30),6); x=rng.normal(size=(360,5))+y[:,None]*.4
    rows=[{"window_id":str(i),"subject":str(groups[i]),"label":int(y[i]),"start_sample":(i%30)*1000,"stop_sample":(i%30+1)*1000,"fs_hz":500} for i in range(360)]
    cal,test=window_masks(rows,config); manifest={"dataset_id":"synthetic"}
    monkeypatch.setattr(runner,"load_data",lambda *args:(x,y,groups,rows,cal,test,config,manifest))
    a,b=tmp_path/"a",tmp_path/"b"
    runner.run(output=a,subjects=["Subject00"])
    x[(groups=="Subject00")&test]+=100
    y[groups=="Subject00"]=1-y[groups=="Subject00"]
    runner.run(output=b,subjects=["Subject00"])
    sa,sb=read_json(a/"Subject00/selection.json"),read_json(b/"Subject00/selection.json")
    for k in ("selected_C","candidates","inner_folds","fit_subjects"): assert sa[k]==sb[k]
    ma,mb=joblib.load(a/"Subject00/models.joblib"),joblib.load(b/"Subject00/models.joblib")
    for key in ma:
        np.testing.assert_array_equal(ma[key]["classifier"].coef_,mb[key]["classifier"].coef_)
        np.testing.assert_array_equal(ma[key]["scaler"].mean_,mb[key]["scaler"].mean_)
        np.testing.assert_array_equal(ma[key]["adapter"].transform_matrix,mb[key]["adapter"].transform_matrix)


@pytest.mark.parametrize("method,expected",[("holm",[.03,.06,.06]),("bh",[.03,.04,.04])])
def test_multiple_comparison_known_values(method,expected):
    np.testing.assert_allclose(adjust_p([.01,.04,.03],method),expected)


def test_sign_exchange_matches_small_exact_null():
    assert sign_flip_p([0,0],repetitions=1000)==1
    assert abs(sign_flip_p([1,1],repetitions=20000)-.5)<.02


def test_whole_subject_label_exchange_preserves_condition_blocks():
    subjects=np.repeat(np.arange(36),60); labels=np.tile(np.repeat([0,1],30),36)
    flips=permutation_flips(42,36,20261009)
    changed=labels^flips[subjects]
    assert set(flips)=={0,1}
    for i in range(36):
        block=changed[subjects==i]
        assert len(set(block[:30]))==1 and len(set(block[30:]))==1 and block[0]!=block[-1]
        assert block.sum()==30
    np.testing.assert_array_equal(permutation_flips(0,36,20261009),np.zeros(36))
