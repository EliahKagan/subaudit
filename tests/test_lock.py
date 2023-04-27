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

"""Tests of locking-related behavior and the ``sub_lock_factory`` argument."""

import contextlib
import enum
import threading
from typing import Generator

import attrs
import mock
from mock import call
import pytest
from pytest_mock import MockerFixture

import subaudit
from tests._helpers import ShortStrEnum
import tests.conftest as ct


@attrs.frozen
class _MockLockFixture:
    """
    Mock lock factory and ``Hook`` that uses it, for the ``mock_lock`` fixture.
    """

    lock_factory: mock.Mock
    """The mock lock (mutex). This does not do any real locking."""

    hook: subaudit.Hook
    """The ``Hook`` instance that was created using the mock lock."""


@enum.unique
class _Scope(ShortStrEnum):
    """
    Ways to supply a mock lock to a ``Hook``: pass locally or patch globally.
    """

    LOCAL = enum.auto()
    """Pass locally."""

    GLOBAL = enum.auto()
    """Patch globally."""


@pytest.fixture(name='mock_lock', params=[_Scope.LOCAL, _Scope.GLOBAL])
def _mock_lock_fixture(
    request: pytest.FixtureRequest, mocker: MockerFixture,
) -> Generator[_MockLockFixture, None, None]:
    """
    A ``Hook`` created with its lock mocked.

    This fixture multiplies tests, covering both a custom ``sub_lock_factory``
    argument and the global ``threading.Lock`.
    """
    lock_factory = mocker.MagicMock(threading.Lock)

    if request.param is _Scope.LOCAL:
        hook = subaudit.Hook(sub_lock_factory=lock_factory)
    elif request.param is _Scope.GLOBAL:
        mocker.patch('threading.Lock', lock_factory)
        hook = subaudit.Hook()
    else:
        # We've exhausted the enumeration, so the "scope" is the wrong type.
        raise TypeError('scope must be a Scope (one of LOCAL or GLOBAL)')

    yield _MockLockFixture(lock_factory=lock_factory, hook=hook)


# pylint: disable=missing-function-docstring  # Tests are descriptively named.


def test_lock_constructed_immediately(mock_lock: _MockLockFixture) -> None:
    """When a ``Hook`` is constructed, it calls its lock factory."""
    mock_lock.lock_factory.assert_called_once_with()


def test_lock_not_entered_immediately(mock_lock: _MockLockFixture) -> None:
    """When a ``Hook`` is constructed, it does not actually enter the lock."""
    mock_lock.lock_factory().__enter__.assert_not_called()


def test_subscribe_enters_and_exits_lock(
    mock_lock: _MockLockFixture, event: str, listener: ct.MockListener,
) -> None:
    mock_lock.hook.subscribe(event, listener)
    calls = mock_lock.lock_factory().mock_calls

    # pylint: disable=unnecessary-dunder-call  # Not really entering/exiting.
    assert calls == [call.__enter__(), call.__exit__(None, None, None)]


def test_unsubscribe_enters_and_exits_lock(
    mock_lock: _MockLockFixture, event: str, listener: ct.MockListener,
) -> None:
    mock_lock.hook.subscribe(event, listener)  # So that we can unsubscribe it.
    mock_lock.lock_factory().reset_mock()  # To only see calls via unsubscribe.
    mock_lock.hook.unsubscribe(event, listener)
    calls = mock_lock.lock_factory().mock_calls

    # pylint: disable=unnecessary-dunder-call  # Not really entering/exiting.
    assert calls == [call.__enter__(), call.__exit__(None, None, None)]


def test_subscribe_never_calls_acquire(
    mock_lock: _MockLockFixture, event: str, listener: ct.MockListener,
) -> None:
    """The lock context manager does not need to have an ``acquire`` method."""
    mock_lock.hook.subscribe(event, listener)
    mock_lock.lock_factory().acquire.assert_not_called()


def test_subscribe_never_calls_release(
    mock_lock: _MockLockFixture, event: str, listener: ct.MockListener,
) -> None:
    """The lock context manager does not need to have a ``release`` method."""
    mock_lock.hook.subscribe(event, listener)
    mock_lock.lock_factory().release.assert_not_called()


def test_unsubscribe_never_calls_acquire(
    mock_lock: _MockLockFixture, event: str, listener: ct.MockListener,
) -> None:
    """The lock context manager does not need to have an ``acquire`` method."""
    mock_lock.hook.subscribe(event, listener)  # So that we can unsubscribe it.
    mock_lock.lock_factory().reset_mock()  # To only see calls via unsubscribe.
    mock_lock.hook.unsubscribe(event, listener)
    mock_lock.lock_factory().acquire.assert_not_called()


def test_unsubscribe_never_calls_release(
    mock_lock: _MockLockFixture, event: str, listener: ct.MockListener,
) -> None:
    """The lock context manager does not need to have a ``release`` method."""
    mock_lock.hook.subscribe(event, listener)  # So that we can unsubscribe it.
    mock_lock.lock_factory().reset_mock()  # To only see calls via unsubscribe.
    mock_lock.hook.unsubscribe(event, listener)
    mock_lock.lock_factory().release.assert_not_called()


@pytest.mark.parametrize('cm_factory', [
    contextlib.nullcontext,
    threading.Lock,
    threading.RLock,
])
def test_lock_accepts_common_cm(
    cm_factory: subaudit.LockContextManagerFactory,
    maybe_raise: ct.MaybeRaiser,
    event: str,
    listener: ct.MockListener,
) -> None:
    """
    A hook with a lock or nullcontext as its lock factory works for listening.

    This is like ``test_listening_observes_only_between_enter_and_exit``, but
    it creates the ``Hook`` with ``contextlib.nullcontext``, ``Lock``, or
    ``RLock`` as its lock factory for subscribing and unsubscribing, as a
    simple but nontrivial check.
    """
    hook = subaudit.Hook(sub_lock_factory=cm_factory)

    subaudit.audit(event, 'a')
    subaudit.audit(event, 'b', 'c')

    with contextlib.suppress(maybe_raise.Exception):
        with hook.listening(event, listener):
            subaudit.audit(event, 'd')
            subaudit.audit(event, 'e', 'f')
            maybe_raise()

    subaudit.audit(event, 'g')
    subaudit.audit(event, 'h', 'i')

    assert listener.mock_calls == [call('d'), call('e', 'f')]


def test_sub_lock_factory_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        # pylint: disable=too-many-function-args  # We are testing that error.
        subaudit.Hook(threading.Lock)  # type: ignore[misc]
