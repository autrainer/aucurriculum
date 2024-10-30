from typing import Type

import pytest

from aucurriculum.curricula.pacing import (
    AbstractPace,
    Exponential,
    Linear,
    Logarithmic,
    OneStep,
    Polynomial,
    Quadratic,
    Root,
)


PACING_FUNCTIONS = [
    Exponential,
    Linear,
    Logarithmic,
    OneStep,
    Polynomial,
    Quadratic,
    Root,
]


class TestPacingFunctions:
    @pytest.mark.parametrize("fn", PACING_FUNCTIONS)
    def test_pacing_size(self, fn: Type[AbstractPace]) -> None:
        pf = fn(0.5, 0.5, 100, 1000)
        assert pf.get_dataset_size(0) == 500, "Should be 500."
        assert pf.get_dataset_size(50) == 1000, "Should be 1000."
        assert pf.get_dataset_size(101) == 1000, "Should be 1000."

    @pytest.mark.parametrize("fn", PACING_FUNCTIONS)
    def test_default_size(self, fn: Type[AbstractPace]) -> None:
        pf = fn(1.0, 0.0, 100, 1000)
        assert pf.get_dataset_size(0) == 1000, "Should be 1000."
        assert pf.get_dataset_size(50) == 1000, "Should be 1000."
        assert pf.get_dataset_size(101) == 1000, "Should be 1000."

    @pytest.mark.parametrize("fn", PACING_FUNCTIONS)
    def test_invalid_initial_size(self, fn: Type[AbstractPace]) -> None:
        with pytest.raises(ValueError):
            fn(0, 0.5, 100, 1000)
        with pytest.raises(ValueError):
            fn(1.1, 0.5, 100, 1000)

    @pytest.mark.parametrize("fn", PACING_FUNCTIONS)
    def test_invalid_final_iteration(self, fn: Type[AbstractPace]) -> None:
        with pytest.raises(ValueError):
            fn(0.5, -0.1, 100, 1000)
        with pytest.raises(ValueError):
            fn(0.5, 1.1, 100, 1000)
