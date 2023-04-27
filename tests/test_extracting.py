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

"""Tests of ``extracting`` context managers."""

import contextlib
from typing import Any, List, Optional, Tuple

import attrs
import mock
from mock import call
import pytest
from pytest_subtests import SubTests
from typing_extensions import Protocol, Self

import subaudit
import tests.conftest as ct


@attrs.frozen
class _Extract:
    """
    Auditing event arguments extracted by a custom extractor.

    The point of this type is to be separate from anything in, or used by, the
    code under test, so no excessively specific behavior wrongly passes tests
    of more general behavior.
    """

    args: Tuple[Any, ...]
    """Event arguments extracted in a test."""

    @classmethod
    def from_separate_args(cls, *args: Any) -> Self:
        """Create an ``_Extract`` instance from separately passed arguments."""
        return cls(args=args)


class _MockExtractor(ct.MockLike, Protocol):
    """
    Protocol for an extractor that supports some of the ``Mock`` interface.
    """

    __slots__ = ()

    def __call__(self, *args: Any) -> _Extract: ...


@pytest.fixture(name='extractor')
def _extractor_fixture() -> _MockExtractor:
    """Mock extractor (pytest fixture). Returns a tuple of its arguments."""
    return mock.Mock(wraps=_Extract.from_separate_args)


# pylint: disable=missing-function-docstring  # Tests are descriptively named.


def test_extracting_does_not_observe_before_enter(
    any_hook: ct.AnyHook, event: str, extractor: _MockExtractor,
) -> None:
    context_manager = any_hook.extracting(event, extractor)
    subaudit.audit(event, 'a', 'b', 'c')
    with context_manager:
        pass
    extractor.assert_not_called()


def test_extracting_does_not_extract_before_enter(
    any_hook: ct.AnyHook, event: str, extractor: _MockExtractor,
) -> None:
    context_manager = any_hook.extracting(event, extractor)
    subaudit.audit(event, 'a', 'b', 'c')
    with context_manager as extracts:
        pass
    assert extracts == []


def test_extracting_does_not_call_subscribe_before_enter(
    derived_hook: ct.DerivedHookFixture, event: str, extractor: _MockExtractor,
) -> None:
    derived_hook.instance.extracting(event, extractor)
    derived_hook.subscribe_method.assert_not_called()


def test_extracting_observes_between_enter_and_exit(
    any_hook: ct.AnyHook, event: str, extractor: _MockExtractor,
) -> None:
    with any_hook.extracting(event, extractor):
        subaudit.audit(event, 'a', 'b', 'c')
    extractor.assert_called_once_with('a', 'b', 'c')


def test_extracting_extracts_between_enter_and_exit(
    any_hook: ct.AnyHook, event: str, extractor: _MockExtractor,
) -> None:
    with any_hook.extracting(event, extractor) as extracts:
        subaudit.audit(event, 'a', 'b', 'c')
    assert extracts == [_Extract(args=('a', 'b', 'c'))]


def test_extracting_enter_calls_subscribe_exactly_once(
    derived_hook: ct.DerivedHookFixture, event: str, extractor: _MockExtractor,
) -> None:
    with derived_hook.instance.extracting(event, extractor):
        derived_hook.subscribe_method.assert_called_once()


def test_extracting_enter_passes_subscribe_same_event_and_hook(
    subtests: SubTests,
    derived_hook: ct.DerivedHookFixture,
    event: str,
    extractor: _MockExtractor,
) -> None:
    with derived_hook.instance.extracting(event, extractor):
        subscribe_hook, subscribe_event, _ = (
            derived_hook.subscribe_method.mock_calls[0].args
        )
        with subtests.test(argument_name='hook'):
            assert subscribe_hook is derived_hook.instance
        with subtests.test(argument_name='event'):
            assert subscribe_event == event


def test_extracting_enter_passes_appender_to_subscribe(
    derived_hook: ct.DerivedHookFixture, event: str, extractor: _MockExtractor,
) -> None:
    with derived_hook.instance.extracting(event, extractor) as extracts:
        subscribe_call, = derived_hook.subscribe_method.mock_calls
        _, _, subscribe_listener = subscribe_call.args

    subscribe_listener('a', 'b', 'c')
    assert extracts == [_Extract(args=('a', 'b', 'c'))]


def test_extracting_does_not_observe_after_exit(
    maybe_raise: ct.MaybeRaiser,
    any_hook: ct.AnyHook,
    event: str,
    extractor: _MockExtractor,
) -> None:
    with contextlib.suppress(maybe_raise.Exception):
        with any_hook.extracting(event, extractor):
            maybe_raise()
    subaudit.audit(event, 'a', 'b', 'c')
    extractor.assert_not_called()


def test_extracting_does_not_extract_after_exit(
    maybe_raise: ct.MaybeRaiser,
    any_hook: ct.AnyHook,
    event: str,
    extractor: _MockExtractor,
) -> None:
    extracts: Optional[List[_Extract]] = None

    with contextlib.suppress(maybe_raise.Exception):
        with any_hook.extracting(event, extractor) as extracts:
            maybe_raise()

    subaudit.audit(event, 'a', 'b', 'c')
    assert extracts == []


def test_extracting_exit_calls_unsubscribe_exactly_once(
    subtests: SubTests,
    maybe_raise: ct.MaybeRaiser,
    derived_hook: ct.DerivedHookFixture,
    event: str,
    extractor: _MockExtractor,
) -> None:
    with contextlib.suppress(maybe_raise.Exception):
        with derived_hook.instance.extracting(event, extractor):
            with subtests.test(where='inside-with-block'):
                derived_hook.unsubscribe_method.assert_not_called()
            maybe_raise()

    with subtests.test(where='after-with-block'):
        derived_hook.unsubscribe_method.assert_called_once()


def test_extracting_exit_passes_unsubscribe_same_event_and_hook(
    subtests: SubTests,
    maybe_raise: ct.MaybeRaiser,
    derived_hook: ct.DerivedHookFixture,
    event: str,
    extractor: _MockExtractor,
) -> None:
    with contextlib.suppress(maybe_raise.Exception):
        with derived_hook.instance.extracting(event, extractor):
            maybe_raise()

    unsubscribe_hook, unsubscribe_event, _ = (
        derived_hook.unsubscribe_method.mock_calls[0].args
    )
    with subtests.test(argument_name='hook'):
        assert unsubscribe_hook is derived_hook.instance
    with subtests.test(argument_name='event'):
        assert unsubscribe_event == event


def test_extracting_exit_passes_appender_to_unsubscribe(
    maybe_raise: ct.MaybeRaiser,
    derived_hook: ct.DerivedHookFixture,
    event: str,
    extractor: _MockExtractor,
) -> None:
    extracts: Optional[List[_Extract]] = None

    with contextlib.suppress(maybe_raise.Exception):
        with derived_hook.instance.extracting(event, extractor) as extracts:
            maybe_raise()

    unsubscribe_call, = derived_hook.unsubscribe_method.mock_calls
    _, _, unsubscribe_listener = unsubscribe_call.args
    unsubscribe_listener('a', 'b', 'c')
    assert extracts == [_Extract(args=('a', 'b', 'c'))]


def test_extracting_observes_only_between_enter_and_exit(
    maybe_raise: ct.MaybeRaiser,
    any_hook: ct.AnyHook,
    event: str,
    extractor: _MockExtractor,
) -> None:
    subaudit.audit(event, 'a')
    subaudit.audit(event, 'b', 'c')

    with contextlib.suppress(maybe_raise.Exception):
        with any_hook.extracting(event, extractor):
            subaudit.audit(event, 'd')
            subaudit.audit(event, 'e', 'f')
            maybe_raise()

    subaudit.audit(event, 'g')
    subaudit.audit(event, 'h', 'i')

    assert extractor.mock_calls == [call('d'), call('e', 'f')]


def test_extracting_extracts_only_between_enter_and_exit(
    maybe_raise: ct.MaybeRaiser,
    any_hook: ct.AnyHook,
    event: str,
    extractor: _MockExtractor,
) -> None:
    extracts: Optional[List[_Extract]] = None

    subaudit.audit(event, 'a')
    subaudit.audit(event, 'b', 'c')

    with contextlib.suppress(maybe_raise.Exception):
        with any_hook.extracting(event, extractor) as extracts:
            subaudit.audit(event, 'd')
            subaudit.audit(event, 'e', 'f')
            maybe_raise()

    subaudit.audit(event, 'g')
    subaudit.audit(event, 'h', 'i')

    assert extracts == [_Extract(args=('d',)), _Extract(args=('e', 'f'))]


def test_extracting_subscribes_and_unsubscribes_same(
    maybe_raise: ct.MaybeRaiser,
    derived_hook: ct.DerivedHookFixture,
    event: str,
    extractor: _MockExtractor,
) -> None:
    with contextlib.suppress(maybe_raise.Exception):
        with derived_hook.instance.extracting(event, extractor):
            maybe_raise()

    subscribe_calls = derived_hook.subscribe_method.mock_calls
    unsubscribe_calls = derived_hook.unsubscribe_method.mock_calls
    assert subscribe_calls == unsubscribe_calls


def test_extracting_delegates_to_listening(
    subtests: SubTests,
    derived_hook: ct.DerivedHookFixture,
    event: str,
    extractor: _MockExtractor,
) -> None:
    """
    Overriding ``listening`` customizes the behavior of ``extracting`` too.
    """
    with derived_hook.instance.extracting(event, extractor):
        with subtests.test('listening called'):
            derived_hook.listening_method.assert_called_once()
        with subtests.test('listening call matches subscribe call'):
            listening_calls = derived_hook.listening_method.mock_calls
            subscribe_calls = derived_hook.subscribe_method.mock_calls
            assert listening_calls == subscribe_calls
