import numpy as np
import pandas as pd

from lungct import data


def _index(n_patients=10, per_patient=27):
    rows = [dict(img_id=f"p{p:02d}_Z_{k:04d}.tif", patient_id=f"p{p:02d}", z_norm=k / (per_patient - 1))
            for p in range(n_patients) for k in range(per_patient)]
    return pd.DataFrame(rows)[:267]


def test_lung_window_is_bounded():
    hu = np.array([-3000.0, -1350.0, -600.0, 150.0, 3000.0])
    x = data.map_intensity(hu, "lung_window")
    assert x.min() == 0.0 and x.max() == 1.0
    assert np.isclose(x[2], 0.5)


def test_hu255_is_linear():
    assert np.allclose(data.map_intensity(np.array([-255.0, 510.0]), "hu255"), [-1.0, 2.0])


def test_random_split_sizes_and_validation():
    df = _index()
    train, test = data.split_indices(df, "random_slice", seed=0)
    train, val = data.validation_split(train, seed=0)
    assert (len(train), len(val), len(test)) == (170, 43, 54)
    assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)


def test_z_blocked_folds_partition_the_slices():
    df = _index()
    tests = [set(data.split_indices(df, "z_blocked", seed=0, z_blocks=4, z_fold=f)[1]) for f in range(4)]
    assert sum(map(len, tests)) == len(df)
    assert set().union(*tests) == set(range(len(df)))


def test_leave_one_patient_out():
    df = _index()
    for seed in range(10):
        train, test = data.split_indices(df, "patient_lopo", seed=seed)
        assert df.patient_id.iloc[test].nunique() == 1
        assert not set(df.patient_id.iloc[train]) & set(df.patient_id.iloc[test])


def test_augmentation_keeps_image_and_mask_aligned():
    rng = np.random.default_rng(0)
    y = (rng.random((32, 32)) > 0.5).astype(np.float32)
    for _ in range(20):
        xa, ya = data.augment(y.copy(), y.copy(), rng)
        assert np.array_equal(xa > 0.5, ya > 0.5)
