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

"""Tests of ``listening`` context managers."""

import contextlib

from mock import call
from pytest_subtests import SubTests

import subaudit
import tests.conftest as ct

# pylint: disable=missing-function-docstring  # Tests are descriptively named.


def test_listening_does_not_observe_before_enter(
    any_hook: ct.AnyHook, event: str, listener: ct.MockListener,
) -> None:
    """The call to ``listening`` does not itself subscribe."""
    context_manager = any_hook.listening(event, listener)
    subaudit.audit(event, 'a', 'b', 'c')
    with context_manager:
        pass
    listener.assert_not_called()


def test_listening_does_not_call_subscribe_before_enter(
    derived_hook: ct.DerivedHookFixture, event: str, listener: ct.MockListener,
) -> None:
    derived_hook.instance.listening(event, listener)
    derived_hook.subscribe_method.assert_not_called()


def test_listening_observes_between_enter_and_exit(
    any_hook: ct.AnyHook, event: str, listener: ct.MockListener,
) -> None:
    """In the block of a ``with``-statement, the listener is subscribed."""
    with any_hook.listening(event, listener):
        subaudit.audit(event, 'a', 'b', 'c')
    listener.assert_called_once_with('a', 'b', 'c')


def test_listening_enter_calls_subscribe(
    derived_hook: ct.DerivedHookFixture, event: str, listener: ct.MockListener,
) -> None:
    """An overridden ``subscribe`` method will be used by ``listening``."""
    with derived_hook.instance.listening(event, listener):
        derived_hook.subscribe_method.assert_called_once_with(
            derived_hook.instance,
            event,
            listener,
        )


def test_listening_does_not_observe_after_exit(
    maybe_raise: ct.MaybeRaiser,
    any_hook: ct.AnyHook,
    event: str,
    listener: ct.MockListener,
) -> None:
    """After exiting the ``with``-statement, the listener is not subscribed."""
    with contextlib.suppress(maybe_raise.Exception):
        with any_hook.listening(event, listener):
            maybe_raise()
    subaudit.audit(event, 'a', 'b', 'c')
    listener.assert_not_called()


def test_listening_exit_calls_unsubscribe(
    subtests: SubTests,
    maybe_raise: ct.MaybeRaiser,
    derived_hook: ct.DerivedHookFixture,
    event: str,
    listener: ct.MockListener,
) -> None:
    """An overridden ``unsubscribe`` method will be called by ``listening``."""
    with contextlib.suppress(maybe_raise.Exception):
        with derived_hook.instance.listening(event, listener):
            with subtests.test(where='inside-with-block'):
                derived_hook.unsubscribe_method.assert_not_called()
            maybe_raise()

    with subtests.test(where='after-with-block'):
        derived_hook.unsubscribe_method.assert_called_once_with(
            derived_hook.instance,
            event,
            listener,
        )


def test_listening_observes_only_between_enter_and_exit(
    maybe_raise: ct.MaybeRaiser,
    any_hook: ct.AnyHook,
    event: str,
    listener: ct.MockListener,
) -> None:
    """
    The ``listening`` context manager works in (simple yet) nontrivial usage.
    """
    subaudit.audit(event, 'a')
    subaudit.audit(event, 'b', 'c')

    with contextlib.suppress(maybe_raise.Exception):
        with any_hook.listening(event, listener):
            subaudit.audit(event, 'd')
            subaudit.audit(event, 'e', 'f')
            maybe_raise()

    subaudit.audit(event, 'g')
    subaudit.audit(event, 'h', 'i')

    assert listener.mock_calls == [call('d'), call('e', 'f')]


def test_listening_enter_returns_none(
    any_hook: ct.AnyHook, event: str, listener: ct.MockListener,
) -> None:
    """The ``listening`` context manager isn't meant to be used with ``as``."""
    with any_hook.listening(event, listener) as context:
        pass
    assert context is None
