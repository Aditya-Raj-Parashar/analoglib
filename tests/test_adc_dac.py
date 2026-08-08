"""Tests for ADC and DAC models."""

import numpy as np
import pytest

from analoglib.adc_dac import ADC, DAC


class TestADC:
    def test_basic_quantization(self):
        adc = ADC(bits=2, v_min=0.0, v_max=3.0)
        # 2 bits → 4 levels: 0, 1, 2, 3
        x = np.array([0.0, 0.4, 1.0, 1.5, 2.0, 2.6, 3.0])
        result = adc.convert(x)
        # Each value should snap to nearest of {0, 1, 2, 3}
        for v in result:
            assert v in (0.0, 1.0, 2.0, 3.0)

    def test_clipping(self):
        adc = ADC(bits=8, v_min=0.0, v_max=1.0)
        x = np.array([-1.0, 0.5, 2.0])
        result = adc.convert(x)
        assert result[0] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)

    def test_resolution(self):
        adc = ADC(bits=8, v_min=0.0, v_max=1.0)
        # 256 levels, step = 1/255
        assert adc.resolution == pytest.approx(1.0 / 255)

    def test_identity_at_levels(self):
        adc = ADC(bits=4, v_min=0.0, v_max=1.0)
        # Values that are exact levels should be unchanged
        levels = np.linspace(0, 1, 16)
        result = adc.convert(levels)
        np.testing.assert_allclose(result, levels, atol=1e-12)

    def test_invalid_bits(self):
        with pytest.raises(ValueError):
            ADC(bits=0)

    def test_invalid_range(self):
        with pytest.raises(ValueError):
            ADC(bits=8, v_min=1.0, v_max=0.0)


class TestDAC:
    def test_basic_quantization(self):
        dac = DAC(bits=2, v_min=0.0, v_max=3.0)
        x = np.array([0.0, 0.4, 1.0, 1.5, 2.0, 2.6, 3.0])
        result = dac.convert(x)
        for v in result:
            assert v in (0.0, 1.0, 2.0, 3.0)

    def test_clipping(self):
        dac = DAC(bits=8, v_min=0.0, v_max=1.0)
        x = np.array([-0.5, 0.5, 1.5])
        result = dac.convert(x)
        assert result[0] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)

    def test_serialization(self):
        dac = DAC(bits=6, v_min=0.0, v_max=3.3)
        d = dac.to_dict()
        assert d["bits"] == 6
        assert d["v_max"] == 3.3
