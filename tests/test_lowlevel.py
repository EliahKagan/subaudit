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

"""Tests of the lower-level ``addaudithook`` and ``audit`` functions."""

import sys

import pytest

import subaudit

# pylint: disable=missing-function-docstring  # Tests are descriptively named.


@pytest.mark.xfail(
    sys.version_info < (3, 8),
    reason='Python 3.8+ has sys.audit.',
    raises=AttributeError,
    strict=True,
)
def test_audit_is_sys_audit_since_3_8() -> None:
    ours = subaudit.audit

    # pylint: disable=no-member
    assert ours is sys.audit  # type: ignore[attr-defined]


@pytest.mark.xfail(
    sys.version_info >= (3, 8),
    reason='Python 3.8+ has sys.audit.',
    raises=ImportError,
    strict=True,
)
def test_audit_is_sysaudit_audit_before_3_8() -> None:
    # pylint: disable=import-error,import-outside-toplevel
    import sysaudit  # type: ignore[import]

    assert subaudit.audit is sysaudit.audit


@pytest.mark.xfail(
    sys.version_info < (3, 8),
    reason='Python 3.8+ has sys.addaudithook.',
    raises=AttributeError,
    strict=True,
)
def test_addaudithook_is_sys_addaudithook_since_3_8() -> None:
    ours = subaudit.addaudithook

    # pylint: disable=no-member
    assert ours is sys.addaudithook  # type: ignore[attr-defined]


@pytest.mark.xfail(
    sys.version_info >= (3, 8),
    reason='Python 3.8+ has sys.addaudithook.',
    raises=ImportError,
    strict=True,
)
def test_addaudithook_is_sysaudit_addaudithook_before_3_8() -> None:
    # pylint: disable=import-error,import-outside-toplevel
    import sysaudit  # type: ignore[import]

    assert subaudit.addaudithook is sysaudit.addaudithook
