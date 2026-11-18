"""
Local automated chain on the Zenodo COVID-19-CT-Seg dataset (record 3757476),
same design as the manuscript (run_local_o1.py + run_local_chain.py):

    O1  faithful recipe, gradient accumulation   20 seeds
    O2  gap decomposition                        4 arms x 12 seeds
    O3  architecture benchmark                   4 archs x 5 seeds
    O6  leakage probe (unet + resunetpp)         random vs z-blocked K=4, 6 seeds
        + leave-one-patient-out, one fold per patient (protocol §Sensitivity:
          true patient labels exist for this dataset)

The literal-batch-10 O1 arm (8 seeds) runs separately:
    $env:LUG_CONFIG = "config_covid.yaml"
    python train_faithful.py --runs 8 --out faithful_truebatch10_all.csv

Prerequisite: 01_data_loading/convert_covid_ct_seg.py has written
<ROOT>/data/{2d_images,2d_masks,dataset_index.csv}.

Resumable: re-launching skips (experiment, seed, ...) rows already in the CSVs.
Results -> <ROOT>/results/
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "colab"))
import lug_method as L  # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "00_inputs", "covid19_ct_seg"))


def banner(msg):
    print(f"\n{'='*70}\n{msg}  |  {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*70}", flush=True)


if __name__ == "__main__":
    assert os.path.exists(os.path.join(ROOT, "data", "dataset_index.csv")), \
        f"run convert_covid_ct_seg.py first (no dataset_index.csv under {ROOT})"
    banner(f"CHAIN START  root={ROOT}")

    banner("O1  faithful recipe  (20 seeds)")
    L.run_faithful(ROOT, n_seeds=20, device="cuda")

    banner("O2  gap decomposition  (4 arms x 12 seeds)")
    L.run_gap_decomposition(ROOT, n_seeds=12, device="cuda")

    banner("O3  architecture benchmark  (4 archs x 5 seeds)")
    L.run_arch_benchmark(ROOT, n_seeds=5, device="cuda")

    banner("O6  leakage probe  (unet + resunetpp: random, z-blocked K=4 x 6 seeds; LOPO)")
    L.run_leakage_probe(ROOT, n_seeds=6, Ks=(4,), device="cuda",
                        archs=("unet", "resunetpp"),
                        schemes=("random_slice", "z_blocked", "patient_lopo"))

    banner("CHAIN DONE")
