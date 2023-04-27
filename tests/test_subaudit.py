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

"""
Tests for the ``subaudit`` module.

These are the tests not yet moved moved into some more specific test module.
"""

# TODO: Finish splitting this into multiple modules.

import contextlib
import datetime
import functools
import io
import pathlib
import platform
import random
import sys
from types import MethodType
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)
import unittest

import attrs
import clock_timer
import mock
from mock import call
import pytest
from pytest_mock import MockerFixture
from pytest_subtests import SubTests
from typing_extensions import Protocol, Self

import subaudit
import tests.conftest as ct


@pytest.fixture(name='equal_listeners', params=[2, 3, 5])
def _equal_listeners_fixture(
    request: pytest.FixtureRequest,
    null_listener: Callable[..., None],
) -> Tuple[ct.MockListener, ...]:
    """Listeners that are different objects but all equal (pytest fixture)."""
    group_key = object()

    def in_group(other: object) -> bool:
        return getattr(other, 'group_key', None) is group_key

    def make_mock() -> ct.MockListener:
        return mock.Mock(
            spec=null_listener,
            __eq__=mock.Mock(side_effect=in_group),
            __hash__=mock.Mock(return_value=hash(group_key)),
            group_key=group_key,
        )

    return tuple(make_mock() for _ in range(request.param))


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


_xfail_no_standard_audit_events_before_3_8 = pytest.mark.xfail(
    sys.version_info < (3, 8),
    reason='Python has no standard audit events before 3.8.',
    raises=AssertionError,
    strict=True,
)
"""
Mark expected failure by ``AssertionError`` due to no library events < 3.8.
"""

# pylint: disable=missing-function-docstring  # Tests are descriptively named.


def test_subscribed_listener_observes_event(
    any_hook: ct.AnyHook, event: str, listener: ct.MockListener,
) -> None:
    any_hook.subscribe(event, listener)
    subaudit.audit(event, 'a', 'b', 'c')
    listener.assert_called_once_with('a', 'b', 'c')


def test_unsubscribed_listener_does_not_observe_event(
    any_hook: ct.AnyHook, event: str, listener: ct.MockListener,
) -> None:
    any_hook.subscribe(event, listener)
    any_hook.unsubscribe(event, listener)
    subaudit.audit(event, 'a', 'b', 'c')
    listener.assert_not_called()


def test_subscribed_listener_does_not_observe_other_event(
    any_hook: ct.AnyHook,
    make_events: ct.MultiSupplier[str],
    listener: ct.MockListener,
) -> None:
    """Subscribing to one event doesn't observe other events."""
    event1, event2 = make_events(2)
    any_hook.subscribe(event1, listener)
    subaudit.audit(event2, 'a', 'b', 'c')
    listener.assert_not_called()


def test_listener_can_subscribe_multiple_events(
    any_hook: ct.AnyHook,
    make_events: ct.MultiSupplier[str],
    listener: ct.MockListener,
) -> None:
    event1, event2 = make_events(2)
    any_hook.subscribe(event1, listener)
    any_hook.subscribe(event2, listener)
    subaudit.audit(event1, 'a', 'b', 'c')
    subaudit.audit(event2, 'd', 'e')
    assert listener.mock_calls == [call('a', 'b', 'c'), call('d', 'e')]


def test_listeners_called_in_subscribe_order(
    any_hook: ct.AnyHook,
    event: str,
    make_listeners: ct.MultiSupplier[ct.MockListener],
) -> None:
    ordering: List[int] = []
    listener1, listener2, listener3 = make_listeners(3)
    listener1.side_effect = functools.partial(ordering.append, 1)
    listener2.side_effect = functools.partial(ordering.append, 2)
    listener3.side_effect = functools.partial(ordering.append, 3)

    any_hook.subscribe(event, listener1)
    any_hook.subscribe(event, listener2)
    any_hook.subscribe(event, listener3)
    subaudit.audit(event)

    assert ordering == [1, 2, 3]


def test_listeners_called_in_subscribe_order_after_others_unsubscribe(
    any_hook: ct.AnyHook,
    event: str,
    make_listeners: ct.MultiSupplier[ct.MockListener],
) -> None:
    ordering: List[int] = []
    listener1, listener2, listener3, listener4 = make_listeners(4)
    listener1.side_effect = functools.partial(ordering.append, 1)
    listener2.side_effect = functools.partial(ordering.append, 2)
    listener3.side_effect = functools.partial(ordering.append, 3)
    listener4.side_effect = functools.partial(ordering.append, 4)

    any_hook.subscribe(event, listener1)
    any_hook.subscribe(event, listener2)
    any_hook.subscribe(event, listener3)
    any_hook.subscribe(event, listener4)
    any_hook.unsubscribe(event, listener1)
    any_hook.unsubscribe(event, listener3)
    subaudit.audit(event)

    assert ordering == [2, 4]


def test_listeners_called_in_new_order_after_resubscribe(
    any_hook: ct.AnyHook,
    event: str,
    make_listeners: ct.MultiSupplier[ct.MockListener],
) -> None:
    ordering: List[int] = []
    listener1, listener2 = make_listeners(2)
    listener1.side_effect = functools.partial(ordering.append, 1)
    listener2.side_effect = functools.partial(ordering.append, 2)

    any_hook.subscribe(event, listener1)
    any_hook.subscribe(event, listener2)
    any_hook.unsubscribe(event, listener1)
    any_hook.subscribe(event, listener1)
    subaudit.audit(event)

    assert ordering == [2, 1]


def test_cannot_unsubscribe_if_never_subscribed(
    any_hook: ct.AnyHook, event: str, listener: ct.MockListener,
) -> None:
    with pytest.raises(ValueError):
        any_hook.unsubscribe(event, listener)


def test_cannot_unsubscribe_if_no_longer_subscribed(
    any_hook: ct.AnyHook, event: str, listener: ct.MockListener,
) -> None:
    any_hook.subscribe(event, listener)
    any_hook.unsubscribe(event, listener)
    with pytest.raises(ValueError):
        any_hook.unsubscribe(event, listener)


@pytest.mark.parametrize('count', [0, 2, 3, 10])
def test_listener_observes_event_as_many_times_as_subscribed(
    count: int, any_hook: ct.AnyHook, event: str, listener: ct.MockListener,
) -> None:
    for _ in range(count):
        any_hook.subscribe(event, listener)
    subaudit.audit(event)
    assert listener.call_count == count


@pytest.mark.parametrize('count', [2, 3, 10])
def test_can_unsubscribe_as_many_times_as_subscribed(
    count: int, any_hook: ct.AnyHook, event: str, listener: ct.MockListener,
) -> None:
    for _ in range(count):
        any_hook.subscribe(event, listener)
    try:
        for _ in range(count):
            any_hook.subscribe(event, listener)
    except ValueError as error:
        pytest.fail(
            f"Couldn't subscribe then unsubscribe {count} times: {error!r}")


@pytest.mark.parametrize('count', [2, 3, 10])
def test_cannot_unsubscribe_more_times_than_subscribed(
    count: int, any_hook: ct.AnyHook, event: str, listener: ct.MockListener,
) -> None:
    for _ in range(count):
        any_hook.subscribe(event, listener)
    with pytest.raises(ValueError):
        for _ in range(count + 1):
            any_hook.unsubscribe(event, listener)


def test_unsubscribe_keeps_other_listener(
    any_hook: ct.AnyHook,
    event: str,
    make_listeners: ct.MultiSupplier[ct.MockListener],
) -> None:
    """Unsubscribing one listener doesn't prevent another from observing."""
    listener1, listener2 = make_listeners(2)
    any_hook.subscribe(event, listener1)
    any_hook.subscribe(event, listener2)
    any_hook.unsubscribe(event, listener1)
    subaudit.audit(event, 'a', 'b', 'c')
    listener2.assert_called_once_with('a', 'b', 'c')


def test_unsubscribe_removes_last_equal_listener(
    any_hook: ct.AnyHook,
    event: str,
    equal_listeners: Tuple[ct.MockListener, ...],
) -> None:
    for listener in equal_listeners:
        any_hook.subscribe(event, listener)
    any_hook.unsubscribe(event, equal_listeners[0])
    subaudit.audit(event, 'a', 'b', 'c')
    equal_listeners[-1].assert_not_called()


def test_unsubscribe_keeps_non_last_equal_listeners(
    subtests: SubTests,
    any_hook: ct.AnyHook,
    event: str,
    equal_listeners: Tuple[ct.MockListener, ...],
) -> None:
    """Unsubscribing removes no equal listeners besides the last subscribed."""
    for listener in equal_listeners:
        any_hook.subscribe(event, listener)
    any_hook.unsubscribe(event, equal_listeners[0])
    subaudit.audit(event, 'a', 'b', 'c')

    for index, listener in enumerate(equal_listeners[:-1]):
        with subtests.test(listener_index=index):
            listener.assert_called_once_with('a', 'b', 'c')


def test_cannot_unsubscribe_listener_from_other_hook(
    make_hooks: ct.MultiSupplier[subaudit.Hook],
    event: str,
    listener: ct.MockListener,
) -> None:
    hook1, hook2 = make_hooks(2)
    hook1.subscribe(event, listener)
    with pytest.raises(ValueError):
        hook2.unsubscribe(event, listener)


def test_instance_construction_does_not_add_audit_hook(
    mocker: MockerFixture,
) -> None:
    """
    ``Hook`` is lazy, not adding an audit hook before a listener subscribes.
    """
    addaudithook = mocker.patch('subaudit.addaudithook')
    subaudit.Hook()
    addaudithook.assert_not_called()


def test_instance_adds_audit_hook_on_first_subscribe(
    mocker: MockerFixture,
    hook: subaudit.Hook,
    event: str,
    listener: ct.MockListener,
) -> None:
    addaudithook = mocker.patch('subaudit.addaudithook')
    hook.subscribe(event, listener)
    addaudithook.assert_called_once()


def test_instance_does_not_add_audit_hook_on_second_subscribe(
    mocker: MockerFixture,
    hook: subaudit.Hook,
    make_events: ct.MultiSupplier[str],
    make_listeners: ct.MultiSupplier[ct.MockListener],
) -> None:
    event1, event2 = make_events(2)
    listener1, listener2 = make_listeners(2)
    hook.subscribe(event1, listener1)
    addaudithook = mocker.patch('subaudit.addaudithook')
    hook.subscribe(event2, listener2)
    addaudithook.assert_not_called()


def test_second_instance_adds_audit_hook_on_first_subscribe(
    mocker: MockerFixture,
    make_hooks: ct.MultiSupplier[subaudit.Hook],
    make_events: ct.MultiSupplier[str],
    make_listeners: ct.MultiSupplier[ct.MockListener],
) -> None:
    """Different ``Hook`` objects do not share the same audit hook."""
    hook1, hook2 = make_hooks(2)
    event1, event2 = make_events(2)
    listener1, listener2 = make_listeners(2)
    hook1.subscribe(event1, listener1)
    addaudithook = mocker.patch('subaudit.addaudithook')
    hook2.subscribe(event2, listener2)
    addaudithook.assert_called_once()


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


@pytest.mark.unstable
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
        elapsed = datetime.timedelta(seconds=timer.total_elapsed)
        assert elapsed <= datetime.timedelta(seconds=8)  # Usually much faster.


@_xfail_no_standard_audit_events_before_3_8
def test_can_listen_to_id_event(
    any_hook: ct.AnyHook, listener: ct.MockListener,
) -> None:
    """
    We can listen to the ``builtins.id`` event.

    See https://docs.python.org/3/library/audit_events.html. We should be able
    to listen to any event listed there, but these tests only try a select few.
    """
    obj = object()
    obj_id = id(obj)
    with any_hook.listening('builtins.id', listener):
        id(obj)
    listener.assert_called_once_with(obj_id)


@_xfail_no_standard_audit_events_before_3_8
def test_can_listen_to_open_event(
    tmp_path: pathlib.Path, any_hook: ct.AnyHook, listener: ct.MockListener,
) -> None:
    """
    We can listen to the ``open`` event.

    See https://docs.python.org/3/library/audit_events.html. We should be able
    to listen to any event listed there, but these tests only try a select few.
    """
    path = tmp_path / 'output.txt'

    with any_hook.listening('open', listener):
        # Using the open builtin instead of path.write_text simplifies the test
        # due to subtleties covered in examples/notebooks/open_event.ipynb.
        with open(path, mode='w', encoding='utf-8'):
            pass

    listener.assert_called_with(str(path), 'w', mock.ANY)


@pytest.mark.xfail(
    platform.python_implementation() == 'CPython',
    reason='CPython only raises builtins.input/result for interactive input.',
    raises=AssertionError,
    strict=True,
)
@_xfail_no_standard_audit_events_before_3_8
def test_can_listen_to_input_events(
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    any_hook: ct.AnyHook,
    make_listeners: ct.MultiSupplier[ct.MockListener],
) -> None:
    """
    We can listen to ``builtins.input`` and ``builtins.input/result`` events.

    See https://docs.python.org/3/library/audit_events.html. We should be able
    to listen to any event listed there, but these tests only try a select few.

    See ``notebooks/input_events.ipynb`` about ``builtins.input/result``
    subtleties.
    """
    prompt = 'What... is the airspeed velocity of an unladen swallow? '
    result = 'What do you mean? An African or European swallow?'
    expected_calls = [call.listen_prompt(prompt), call.listen_result(result)]

    parent = mock.Mock()  # To assert calls to child mocks in a specific order.
    parent.listen_prompt, parent.listen_result = make_listeners(2)

    with any_hook.listening('builtins.input', parent.listen_prompt):
        with any_hook.listening('builtins.input/result', parent.listen_result):
            with monkeypatch.context() as context:
                context.setattr(sys, 'stdin', io.StringIO(result))
                returned_result = input(prompt)

    written_prompt = capsys.readouterr().out
    if written_prompt != prompt:
        raise RuntimeError(f'got output {written_prompt!r}, need {prompt!r}')
    if returned_result != result:
        raise RuntimeError(f'got input {returned_result!r}, need {result!r}')

    assert parent.mock_calls == expected_calls


def test_can_listen_to_addaudithook_event(
    make_hooks: ct.MultiSupplier[subaudit.Hook],
    event: str,
    make_listeners: ct.MultiSupplier[ct.MockListener],
) -> None:
    """
    We can listen to the ``sys.addaudithook`` event.

    See https://docs.python.org/3/library/audit_events.html. We should be able
    to listen to any event listed there, but these tests only try a select few.

    The ``sysaudit`` library backports this event to Python 3.7.
    """
    hook1, hook2 = make_hooks(2)
    listener1, listener2 = make_listeners(2)
    with hook1.listening('sys.addaudithook', listener1):
        with hook2.listening(event, listener2):
            listener1.assert_called_once_with()
