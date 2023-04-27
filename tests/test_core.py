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
Tests of core ``subaudit`` functionality.

This includes tests of ``subscribe`` and ``unsubscribe``, except how they lock,
which is tested in ``test_lock.py`` instead.

This also includes tests of ``Hook`` construction, specifically that creating a
``Hook`` instance does not install an audit hook before the first subscription.
Tests of the ``sub_lock_factory`` parameter on construction are also in
``test_lock.py`` rather that here.

Note that the core functionality should not be confused with the functionality
that code using the ``subaudit`` library is most likely to use directly. The
best way to use this library is often to use ``subaudit.listening``. See
``test_listening.py`` (and ``test_misc.py``).
"""

import functools
from typing import Callable, List, Tuple

import mock
from mock import call
import pytest
from pytest_mock import MockerFixture
from pytest_subtests import SubTests

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
