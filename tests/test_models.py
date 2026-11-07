import pytest
import torch

from lungct.models import ARCHITECTURES, build_model, count_parameters

EXPECTED_PARAMETERS = {"unet": 7_765_409, "attn_unet": 7_853_173, "resunet": 8_115_073, "resunetpp": 4_085_873}


@pytest.mark.parametrize("name", list(ARCHITECTURES))
def test_parameter_count(name):
    assert count_parameters(build_model(name)) == EXPECTED_PARAMETERS[name]


@pytest.mark.parametrize("name", list(ARCHITECTURES))
def test_output_is_a_probability_map(name):
    model = build_model(name).eval()
    with torch.no_grad():
        y = model(torch.randn(1, 1, 64, 64))
    assert y.shape == (1, 1, 64, 64)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
