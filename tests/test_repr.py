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

"""Tests for the ``repr``s of ``Hook`` objects."""

import re

import attrs
import pytest

import subaudit
import tests.conftest as ct


@attrs.frozen
class _ReprAsserter:
    """Callable to assert correct ``Hook`` repr."""

    def __call__(
        self,
        hook: subaudit.Hook,
        summary_pattern: str,
        *,
        type_name_pattern: str = 'Hook',
    ) -> None:
        """Assert the repr is the non-code style with a matching summary."""
        regex = rf'<{type_name_pattern} at 0x[0-9a-fA-F]+: {summary_pattern}>'
        assert re.fullmatch(regex, repr(hook))


@pytest.fixture(name='assert_repr_summary')
def _assert_repr_summary_fixture() -> _ReprAsserter:
    """
    Callable to assert a ``Hook`` instance has a correct repr.

    This fixture provides an object that, when called with a ``Hook`` instance
    and the pattern that should match a summary of its state, asserts that the
    instance's repr is in the non-code style (with ``<`` ``>``) and contains
    the usual general information followed by a summary matching the pattern.
    """
    return _ReprAsserter()


# pylint: disable=missing-function-docstring  # Tests are descriptively named.


def test_repr_shows_hook_not_installed_on_creation(
    hook: subaudit.Hook, assert_repr_summary: _ReprAsserter,
) -> None:
    """A new ``Hook``'s repr has general info and the not-installed summary."""
    assert_repr_summary(hook, r'audit hook not installed')


def test_repr_shows_one_event_while_listening(
    hook: subaudit.Hook,
    event: str,
    listener: ct.MockListener,
    assert_repr_summary: _ReprAsserter,
) -> None:
    with hook.listening(event, listener):
        assert_repr_summary(hook, r'watching 1 event')


def test_repr_shows_no_events_after_done_listening(
    hook: subaudit.Hook,
    event: str,
    listener: ct.MockListener,
    assert_repr_summary: _ReprAsserter,
) -> None:
    with hook.listening(event, listener):
        pass
    assert_repr_summary(hook, r'watching 0 events')


def test_repr_shows_both_events_when_nested_listening(
    hook: subaudit.Hook,
    make_events: ct.MultiSupplier[str],
    listener: ct.MockListener,
    assert_repr_summary: _ReprAsserter,
) -> None:
    event1, event2 = make_events(2)
    with hook.listening(event1, listener):
        with hook.listening(event2, listener):
            assert_repr_summary(hook, r'watching 2 events')


def test_repr_shows_one_event_after_done_listening_to_second(
    hook: subaudit.Hook,
    make_events: ct.MultiSupplier[str],
    listener: ct.MockListener,
    assert_repr_summary: _ReprAsserter,
) -> None:
    event1, event2 = make_events(2)
    with hook.listening(event1, listener):
        with hook.listening(event2, listener):
            pass
        assert_repr_summary(hook, r'watching 1 event')


def test_repr_shows_no_events_after_done_nested_listening(
    hook: subaudit.Hook,
    make_events: ct.MultiSupplier[str],
    listener: ct.MockListener,
    assert_repr_summary: _ReprAsserter,
) -> None:
    event1, event2 = make_events(2)
    with hook.listening(event1, listener):
        with hook.listening(event2, listener):
            pass
    assert_repr_summary(hook, r'watching 0 events')


def test_repr_shows_one_event_with_multiple_listeners_as_one(
    hook: subaudit.Hook,
    event: str,
    make_listeners: ct.MultiSupplier[ct.MockListener],
    assert_repr_summary: _ReprAsserter,
) -> None:
    listener1, listener2 = make_listeners(2)
    with hook.listening(event, listener1):
        with hook.listening(event, listener2):
            assert_repr_summary(hook, r'watching 1 event')


def test_repr_uses_derived_class_type_name(
        assert_repr_summary: _ReprAsserter,
) -> None:
    class MyHook(subaudit.Hook):
        """Derived class for repr testing."""

    assert_repr_summary(
        MyHook(),
        r'audit hook not installed',
        type_name_pattern=r'MyHook',
    )
