import numpy as np
import pytest

from fishweight.allometric import AllometricRegressor

AREA = np.array([1000.0, 2000, 4000, 8000, 16000, 32000])


def test_recovers_exact_power_law():
    model = AllometricRegressor().fit(AREA, 0.02 * AREA**1.5)
    assert model.a_ == pytest.approx(0.02)
    assert model.b_ == pytest.approx(1.5)
    assert model.predict(np.array([5000.0])) == pytest.approx([0.02 * 5000**1.5])


def test_per_species_coefficients():
    area = np.concatenate([AREA, AREA])
    species = np.array(["x"] * 6 + ["y"] * 6)
    weight = np.where(species == "x", 0.02, 0.05) * area**1.5
    model = AllometricRegressor().fit(area, weight, species)
    assert model.predict(np.array([5000.0, 5000.0]), np.array(["x", "y"])) == pytest.approx(
        [0.02 * 5000**1.5, 0.05 * 5000**1.5])


def test_unseen_or_rare_species_falls_back_to_global():
    area = np.concatenate([AREA, AREA[:2]])
    species = np.array(["x"] * 6 + ["rara"] * 2)
    model = AllometricRegressor().fit(area, 0.02 * area**1.5, species)
    assert "rara" not in model.per_species_
    assert model.predict(np.array([5000.0, 5000.0]), np.array(["rara", "nunca_vista"])) == pytest.approx(
        [0.02 * 5000**1.5] * 2)


def test_non_positive_area_is_rejected():
    with pytest.raises(ValueError):
        AllometricRegressor().fit(np.array([0.0, 10.0]), np.array([1.0, 2.0]))
