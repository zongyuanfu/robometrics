"""Missing measurements must not look like measured zero error or runtime."""
from dataclasses import astuple

import numpy as np
import pytest

from robometrics.statistics import Statistic, TrajectoryGroupMetrics, TrajectoryMetrics


@pytest.mark.parametrize('samples', [[], [np.inf], [np.nan, np.inf, -np.inf]])
def test_no_finite_samples_are_reported_as_missing(samples):
    assert all(np.isnan(value) for value in astuple(Statistic.from_list(samples)))


def test_statistics_ignore_nonfinite_measurements_and_accept_numpy():
    actual = Statistic.from_list(np.array([1., np.nan, 3., np.inf, -np.inf]))
    expected = Statistic.from_list([1., 3.])
    assert actual == expected
    assert actual.mean == 2. and actual.median == 2. and actual.std == 1.


def test_all_failed_group_has_no_measured_success_runtime():
    summary = TrajectoryGroupMetrics.from_list([TrajectoryMetrics()])
    assert summary.success == 0 and summary.skips == 1
    assert np.isnan(summary.solve_time.mean) and np.isnan(summary.position_error.mean)


def test_success_with_missing_runtime_does_not_crash():
    summary = TrajectoryGroupMetrics.from_list([TrajectoryMetrics(skip=False, success=True)])
    assert np.isnan(summary.solve_time.mean)


def test_statistics_reject_nested_samples():
    with pytest.raises(ValueError, match='one-dimensional'):
        Statistic.from_list([[1, 2]])
