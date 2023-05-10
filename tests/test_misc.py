# Copyright (c) 2023 Eliah Kagan
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.

"""Bikeshed for tests not placed in one of the more specific test modules."""

import atexit
import datetime
import functools
import logging
import platform
import random
from types import MethodType
from typing import List, Sequence, cast

import attrs
import clock_timer
from pprint import pformat
import pytest
from pytest_subtests import SubTests

import subaudit
import tests.conftest as ct

_logger = logging.getLogger(__name__)
"""Logger for this test module. Used to report details about churn tests."""


@pytest.mark.implementation_detail
def test_top_level_functions_are_bound_methods(subtests: SubTests) -> None:
    """The module-level functions are bound methods of a ``Hook`` object."""
    top_level_functions = cast(Sequence[MethodType], [
        subaudit.subscribe,
        subaudit.unsubscribe,
        subaudit.listening,
        subaudit.extracting,
    ])

    for func in top_level_functions:
        with subtests.test('bound to a hook', name=func.__name__):
            assert isinstance(func.__self__, subaudit.Hook)

    with subtests.test('all bound to the same one'):
        assert len({func.__self__ for func in top_level_functions}) == 1


@attrs.frozen
class _ChurnCounts:
    """Parameters for a churn test. (``test_usable_in_high_churn`` helper.)"""

    listeners: int
    """Total number of listeners."""

    delta: int
    """Number of listeners unsubscribed and resubscribed per test iteration."""

    iterations: int
    """Number of test iterations."""


@attrs.frozen
class _ChurnReport:
    """Report from a run of the churn test."""

    counts: _ChurnCounts
    """The test's count parameters."""

    elapsed: datetime.timedelta
    """The test's elapsed time."""


_churn_reports: List[_ChurnReport] = []
"""Reports from the churn tests."""


@atexit.register
def _output_churn_reports() -> None:
    """Print some info from ``_churn_reports``, depending on logging level."""
    if not _churn_reports:
        return

    logging.basicConfig(level=logging.DEBUG)  # FIXME: Remove after debugging.

    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug('Churn reports: %s', pformat(_churn_reports))

    if _logger.isEnabledFor(logging.INFO):
        deltas = (report.elapsed for report in _churn_reports)
        total = sum(deltas, start=datetime.timedelta(0))
        _logger.info('Churn total time: %s', total)


@pytest.mark.slow
def test_usable_in_high_churn(
    subtests: SubTests,
    hook: subaudit.Hook,
    event: str,
    make_listeners: ct.MultiSupplier[ct.MockListener],
) -> None:
    """
    ~1000 listeners with frequent ``subscribe``/``unsubscribe`` isn't too slow.
    """
    counts = _ChurnCounts(listeners=1000, delta=100, iterations=100)
    all_listeners = make_listeners(count=counts.listeners)
    prng = random.Random(18140838203929040771)

    all_expected_observations: List[List[int]] = []
    all_observations: List[List[int]] = []
    observations: List[int] = []

    for number, listener in enumerate(all_listeners):
        listener.side_effect = functools.partial(observations.append, number)
        hook.subscribe(event, listener)

    attached = list(range(counts.listeners))

    with clock_timer.ClockLogger() as timer:
        for _ in range(counts.iterations):
            detached: List[int] = []

            for _ in range(counts.delta):
                number = attached.pop(prng.randrange(len(attached)))
                hook.unsubscribe(event, all_listeners[number])
                detached.append(number)

            while detached:
                number = detached.pop(prng.randrange(len(detached)))
                hook.subscribe(event, all_listeners[number])
                attached.append(number)

            all_expected_observations.append(attached[:])
            subaudit.audit(event)
            all_observations.append(observations[:])
            observations.clear()

    with subtests.test('listener calls in correct order'):
        assert all_observations == all_expected_observations

    with subtests.test('elapsed time not excessive'):
        if platform.python_implementation == 'CPython':
            threshold = datetime.timedelta(seconds=4)
        else:
            threshold = datetime.timedelta(seconds=11)

        elapsed = datetime.timedelta(seconds=timer.total_elapsed)
        _churn_reports.append(_ChurnReport(counts, elapsed))
        assert elapsed <= threshold
