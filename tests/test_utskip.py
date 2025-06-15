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

"""Tests of the ``@skip_if_unavailable`` decorator."""

import sys
import unittest

import mock
import pytest
from pytest_check.context_manager import CheckContextManager

import subaudit

# pylint: disable=missing-function-docstring  # Tests are descriptively named.


@pytest.mark.xfail(
    sys.version_info >= (3, 8),
    reason='Python 3.8 has PEP 578, so @skip_if_unavailable should NOT skip.',
    raises=pytest.fail.Exception,
    strict=True,
)
def test_skip_if_unavailable_skips_before_3_8() -> None:
    wrapped = mock.Mock(wraps=lambda: None)
    wrapper = subaudit.skip_if_unavailable(wrapped)
    with pytest.raises(unittest.SkipTest):
        wrapper()


@pytest.mark.xfail(
    sys.version_info < (3, 8),
    reason='Python < 3.8 lacks PEP 578, so @skip_if_unavailable SHOULD skip.',
    raises=AssertionError,
    strict=True,
)
def test_skip_if_unavailable_does_not_skip_since_3_8(
    check: CheckContextManager,  # Works better than subtests, due to xfail.
) -> None:
    wrapped = mock.Mock(wraps=lambda: None)
    wrapper = subaudit.skip_if_unavailable(wrapped)

    with check('SkipTest should not be raised'):
        try:
            wrapper()
        except unittest.SkipTest as skip_exception:
            # Intercept SkipTest, or THIS pytest test will show as skipped!
            # (Use AssertionError instead of calling pytest.fail, to play well
            # with pytest-check. This also simplifies the xfail decoration.)
            message = f'SkipTest exception wrongly raised: {skip_exception!r}'
            raise AssertionError(message) from skip_exception

    with check('The "wrapped" function should be called'):
        wrapped.assert_called_once_with()


def test_zzzfail():
    assert False
