import numpy as np
import torch

from lungct.data import SliceDataset
from lungct.training import _optimizer_steps


def _dataset(n):
    return SliceDataset(np.zeros((n, 8, 8), np.float32), np.zeros((n, 8, 8), np.float32))


def test_batch_of_ten_uses_micro_batches_4_4_2():
    torch.manual_seed(0)
    steps = list(_optimizer_steps(_dataset(170), batch=10, micro_batch=4))
    assert len(steps) == 17
    assert all([len(x) for x, _, _ in s] == [4, 4, 2] for s in steps)


def test_weights_average_the_loss_over_each_step():
    torch.manual_seed(0)
    for s in _optimizer_steps(_dataset(161), batch=8, micro_batch=4):
        assert abs(sum(w for _, _, w in s) - 1.0) < 1e-9


def test_every_slice_used_once_per_epoch():
    torch.manual_seed(0)
    n = sum(len(x) for s in _optimizer_steps(_dataset(213), batch=10, micro_batch=4) for x, _, _ in s)
    assert n == 213
