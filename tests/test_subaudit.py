"""Tests for the subaudit module."""

import contextlib
import functools
import sys
from typing import Any, Callable, Iterator, TypeVar
from unittest.mock import Mock, call
import uuid

import attrs
import pytest
from pytest import FixtureRequest
from pytest_subtests import SubTests

import subaudit
from subaudit import Hook

_T = TypeVar('_T')


class _UnboundMethodMock(Mock):
    """A mock that is also a descriptor, to behave like a function."""

    def __get__(self, instance, owner=None):
        """Bind the instance, if any, to produce a "bound method"."""
        return self if instance is None else functools.partial(self, instance)


class _FakeError(Exception):
    """Fake exception for testing."""


@pytest.fixture(name='maybe_raise', params=[False, True])
def _maybe_raise(request: FixtureRequest) -> Callable[[], None]:
    """
    A function that, when called, either raises _FakeError or does nothing.

    This parameterized fixture multiplies tests that use it, covering both the
    raising and non-raising cases.
    """
    def maybe_raise_now() -> None:
        if request.param:
            raise _FakeError

    return maybe_raise_now


def _generate(supplier: Callable[[], _T]) -> Iterator[_T]:
    """Yield indefinitely many elements, each by calling the supplier."""
    while True:
        yield supplier()


def _make_hook() -> Hook:
    """Create a hook instance."""
    return Hook()


@pytest.fixture(name='hook')
def _hook() -> Hook:
    """Hook instance."""
    return _make_hook()


@pytest.fixture(name='some_hooks')
def _some_hooks() -> Iterator[Hook]:
    """Iterator that gives as many Hook instances as needed."""
    return _generate(_make_hook)


@pytest.fixture(name='mocked_subscribe_unsubscribe_cls')
def _mocked_subscribe_unsubscribe_hook_cls() -> type[Hook]:
    """New Hook subclass with mocked out subscribe and unsubscribe methods."""
    class MockedSubscribeUnsubscribeHook(Hook):
        subscribe = _UnboundMethodMock()
        unsubscribe = _UnboundMethodMock()

    return MockedSubscribeUnsubscribeHook


def _make_event() -> str:
    """Create a randomly generated fake event name."""
    return f'test-subaudit-{uuid.uuid4()}'


@pytest.fixture(name='event')
def _event() -> str:
    """Randomly generated fake event name."""
    return _make_event()


@pytest.fixture(name='some_events')
def _some_events() -> Iterator[str]:
    """Iterator that gives as many fake event names as needed."""
    return _generate(_make_event)


def _make_listener() -> Mock:
    """Create a mock listener."""
    return Mock()


@pytest.fixture(name='listener')
def _listener() -> Mock:
    """Mock listener."""
    return _make_listener()


@pytest.fixture(name='some_listeners')
def _some_listeners() -> Iterator[Mock]:
    """Iterator that gives as many mock listeners as needed."""
    return _generate(_make_listener)


@pytest.fixture(name='nonidentical_equal_listeners', params=[2, 3, 5])
def _nonidentical_equal_listeners(request: FixtureRequest) -> list[Mock]:
    """List of listeners that are different objects but all equal."""
    group_key = object()

    def in_group(other: object) -> bool:
        try:
            return other.group_key is group_key
        except AttributeError:
            return NotImplemented

    def make_mock() -> Mock:
        return Mock(
            __eq__=Mock(side_effect=in_group),
            __hash__=Mock(return_value=hash(group_key)),
            group_key=group_key,
        )

    return [make_mock() for _ in range(request.param)]


@attrs.frozen
class _Extract:
    """
    Auditing event arguments extracted by a custom extractor.

    The point of this type is to be separate from anything in, or used by, the
    code under test, so no excessively specific behavior wrongly passes tests
    of more general behavior.
    """
    args: tuple[Any, ...]
    """Event arguments extracted in a test."""


@pytest.fixture(name='extractor')
def _extractor() -> Mock:
    """Mock extractor. Returns a tuple of its arguments."""
    return Mock(side_effect=lambda *args: _Extract(args))


# FIXME: Change each skipif to xfail with a condition.
#        Use a raises argument where appropriate.


@pytest.mark.skipif(
    sys.version_info < (3, 8),
    reason="Python 3.8+ has sys.audit",
)
def test_audit_is_sys_audit_since_3_8() -> None:
    assert subaudit.audit is sys.audit


@pytest.mark.skipif(
    sys.version_info >= (3, 8),
    reason="Python 3.8+ has sys.audit",
)
def test_audit_is_sysaudit_audit_before_3_8() -> None:
    import sysaudit
    assert subaudit.audit is sysaudit.audit


@pytest.mark.skipif(
    sys.version_info < (3, 8),
    reason="Python 3.8+ has sys.addaudithook",
)
def test_addaudithook_is_sys_addaudithook_since_3_8() -> None:
    assert subaudit.addaudithook is sys.addaudithook


@pytest.mark.skipif(
    sys.version_info >= (3, 8),
    reason="Python 3.8+ has sys.addaudithook",
)
def test_addaudithook_is_sysaudit_addaudithook_before_3_8() -> None:
    import sysaudit
    assert subaudit.addaudithook is sysaudit.addaudithook


def test_subscribed_listener_observes_event(
    hook: Hook, event: str, listener: Mock,
) -> None:
    hook.subscribe(event, listener)
    subaudit.audit(event, 'a', 'b', 'c')
    listener.assert_called_once_with('event', 'a', 'b', 'c')


def test_unsubscribed_listener_does_not_observe_event(
        hook: Hook, event: str, listener: Mock,
) -> None:
    hook.subscribe(event, listener)
    hook.unsubscribe(event, listener)
    subaudit.audit(event, 'a', 'b', 'c')
    listener.assert_not_called()


def test_subscribed_listener_does_not_observe_other_event(
    hook: Hook, some_events: Iterator[str], listener: Mock,
) -> None:
    """Subscribing to one event doesn't observe other events."""
    event1, event2 = some_events
    hook.subscribe(event1, listener)
    subaudit.audit(event2, 'a', 'b', 'c')
    listener.assert_not_called()


def test_listener_can_subscribe_multiple_events(
    hook: Hook, some_events: Iterator[str], listener: Mock,
) -> None:
    event1, event2 = some_events
    hook.subscribe(event1)
    hook.subscribe(event2)
    subaudit.audit(event1, 'a', 'b', 'c')
    subaudit.audit(event2, 'd', 'e')
    assert listener.mock_calls == [call('a', 'b', 'c'), call('d', 'e')]


def test_listeners_called_in_subscribe_order(
    hook: Hook, event: str, some_listeners: Iterator[Mock],
) -> None:
    ordering = []
    listener1, listener2, listener3 = some_listeners
    listener1.side_effect = functools.partial(ordering.append, 1)
    listener2.side_effect = functools.partial(ordering.append, 2)
    listener3.side_effect = functools.partial(ordering.append, 3)

    hook.subscribe(event, listener1)
    hook.subscribe(event, listener2)
    hook.subscribe(event, listener3)
    subaudit.audit(event)

    assert ordering == [1, 2, 3]


def test_listeners_called_in_subscribe_order_after_others_unsubscribe(
    hook: Hook, event: str, some_listeners: Iterator[Mock],
) -> None:
    ordering = []
    listener1, listener2, listener3, listener4 = some_listeners
    listener1.side_effect = functools.partial(ordering.append, 1)
    listener2.side_effect = functools.partial(ordering.append, 2)
    listener3.side_effect = functools.partial(ordering.append, 3)
    listener3.side_effect = functools.partial(ordering.append, 4)

    hook.subscribe(event, listener1)
    hook.subscribe(event, listener2)
    hook.subscribe(event, listener3)
    hook.subscribe(event, listener4)
    hook.unsubscribe(event, listener1)
    hook.unsubscribe(event, listener3)
    subaudit.audit(event)

    assert ordering == [2, 4]


def test_listeners_called_in_new_order_after_resubscribe(
    hook: Hook, event: str, some_listeners: Iterator[Mock],
) -> None:
    ordering = []
    listener1, listener2 = some_listeners
    listener1.side_effect = functools.partial(ordering.append, 1)
    listener2.side_effect = functools.partial(ordering.append, 2)

    hook.subscribe(event, listener1)
    hook.subscribe(event, listener2)
    hook.unsubscribe(event, listener1)
    hook.subscribe(event, listener1)
    subaudit.audit(event)

    assert ordering == [2, 1]


def test_cannot_unsubscribe_if_never_subscribed(
    hook: Hook, event: str, listener: Mock,
) -> None:
    with pytest.raises(ValueError):
        hook.unsubscribe(event, listener)


def test_cannot_unsubscribe_if_no_longer_subscribed(
    hook: Hook, event: str, listener: Mock,
) -> None:
    hook.subscribe(event, listener)
    hook.unsubscribe(event, listener)
    with pytest.raises(ValueError):
        hook.unsubscribe(event, listener)


@pytest.mark.parametrize('count', [0, 2, 3, 10])
def test_listener_observes_event_as_many_times_as_subscribed(
    count: int, hook: Hook, event: str, listener: Mock,
) -> None:
    for _ in range(count):
        hook.subscribe(event, listener)
    subaudit.audit(event)
    assert listener.call_count == count


@pytest.mark.parametrize('count', [2, 3, 10])
def test_can_unsubscribe_as_many_times_as_subscribed(
    count: int, hook: Hook, event: str, listener: Mock,
) -> None:
    for _ in range(count):
        hook.subscribe(event, listener)
    try:
        for _ in range(count):
            hook.subscribe(event, listener)
    except ValueError as error:
        pytest.fail(
            f"Couldn't subscribe then unsubscribe {count} times: {error!r}")


@pytest.mark.parametrize('count', [2, 3, 10])
def test_cannot_unsubscribe_more_times_than_subscribed(
    count: int, hook: Hook, event: str, listener: Mock,
) -> None:
    for _ in range(count):
        hook.subscribe(event, listener)
    with pytest.raises(ValueError):
        for _ in range(count + 1):
            hook.subscribe(event, listener)


def test_unsubscribe_keeps_other_listener(
    hook: Hook, event: str, some_listeners: Iterator[Mock],
) -> None:
    """Unsubscribing one listener doesn't prevent another from observing."""
    listener1, listener2 = some_listeners
    hook.subscribe(event, listener1)
    hook.subscribe(event, listener2)
    hook.unsubscribe(event, listener1)
    subaudit.audit(event, 'a', 'b', 'c')
    listener2.assert_called_once_with('a', 'b', 'c')


def test_unsubscribe_removes_last_equal_listener(
    hook: Hook, event: str, nonidentical_equal_listeners: list[Mock],
) -> None:
    for listener in nonidentical_equal_listeners:
        hook.subscribe(event, listener)
    hook.unsubscribe(event, nonidentical_equal_listeners[0])
    subaudit.audit(event, 'a', 'b', 'c')
    nonidentical_equal_listeners[-1].assert_not_called()


def test_unsubscribe_keeps_non_last_equal_listeners(
    subtests: SubTests,
    hook: Hook,
    event: str,
    nonidentical_equal_listeners: list[Mock],
) -> None:
    """Unsubscribing removes no equal listeners besides the last subscribed."""
    for listener in nonidentical_equal_listeners:
        hook.subscribe(event, listener)
    hook.unsubscribe(event, nonidentical_equal_listeners[0])
    subaudit.audit(event, 'a', 'b', 'c')

    for index, listener in enumerate(nonidentical_equal_listeners[:-1]):
        with subtests.test(listener_index=index):
            listener.assert_called_once_with('a', 'b', 'c')


def test_cannot_unsubscribe_listener_from_other_hook(
    some_hooks: Iterator[Hook], event: str, listener: Mock,
) -> None:
    hook1, hook2 = some_hooks
    hook1.subscribe(event, listener)
    with pytest.raises(ValueError):
        hook2.unsubscribe(event, listener)


# FIXME: Test that a Hook's first subscription adds an audit hook.


# FIXME: Test that a Hook's subsequent subscriptions don't add audit hooks.


# FIXME: Test that a second Hook's first subscription adds an audit hook.


def test_listening_does_not_observe_before_enter(
    hook: Hook, event: str, listener: Mock,
) -> None:
    """The call to listening does not itself subscribe."""
    context_manager = hook.listening(event, listener)
    subaudit.audit(event, 'a', 'b', 'c')
    with context_manager:
        pass
    listener.assert_not_called()


def test_listening_does_not_call_subscribe_before_enter(
    mocked_subscribe_unsubscribe_cls: type[Hook], event: str, listener: Mock,
) -> None:
    subscribe = mocked_subscribe_unsubscribe_cls.subscribe
    hook = mocked_subscribe_unsubscribe_cls()
    hook.listening(event, listener)
    subscribe.assert_not_called()


def test_listening_observes_between_enter_and_exit(
    hook: Hook, event: str, listener: Mock,
) -> None:
    """In the block of a with statement, the listener is subscribed."""
    with hook.listening(event, listener):
        subaudit.audit(event, 'a', 'b', 'c')
    listener.assert_called_once_with('a', 'b', 'c')


def test_listening_enter_calls_subscribe(
    mocked_subscribe_unsubscribe_cls: type[Hook], event: str, listener: Mock,
) -> None:
    """An overridden subscribe method will be used by listening."""
    subscribe = mocked_subscribe_unsubscribe_cls.subscribe
    hook = mocked_subscribe_unsubscribe_cls()
    with hook.listening(event, listener):
        subscribe.assert_called_once_with(hook, event, listener)


def test_listening_does_not_observe_after_exit(
    maybe_raise: Callable[[], None], hook: Hook, event: str, listener: Mock,
) -> None:
    """After exiting the with statement, the listener is not subscribed."""
    with contextlib.suppress(_FakeError):
        with hook.listening(event, listener):
            maybe_raise()
    subaudit.audit(event, 'a', 'b', 'c')
    listener.assert_not_called()


def test_listening_exit_calls_unsubscribe(
    maybe_raise: Callable[[], None],
    mocked_subscribe_unsubscribe_cls: type[Hook],
    event: str,
    listener: Mock,
) -> None:
    """An overridden unsubscribe method will be called by listening."""
    unsubscribe = mocked_subscribe_unsubscribe_cls.unsubscribe
    hook = mocked_subscribe_unsubscribe_cls()

    with contextlib.suppress(_FakeError):
        with hook.listening(event, listener):
            maybe_raise()

    unsubscribe.assert_called_once_with(hook, event, listener)


def test_listening_observes_only_between_enter_and_exit(
    maybe_raise: Callable[[], None], hook: Hook, event: str, listener: Mock,
) -> None:
    """The listening context manager works in (simple yet) nontrivial usage."""
    subaudit.audit(event, 'a')
    subaudit.audit(event, 'b', 'c')

    with contextlib.suppress(_FakeError):
        with hook.listening(event, listener):
            subaudit.audit('d')
            subaudit.audit('e', 'f')
            maybe_raise()

    subaudit.audit(event, 'g')
    subaudit.audit(event, 'h', 'i')

    assert listener.mock_calls == [call('d'), call('e', 'f')]


def test_listening_enter_returns_none(
    hook: Hook, event: str, listener: Mock,
) -> None:
    """The listening context manager isn't meant to be used with "as"."""
    with hook.listening(event, listener) as context:
        pass
    assert context is None


def test_extracting_does_not_observe_before_enter(
    hook: Hook, event: str, extractor: Mock,
) -> None:
    context_manager = hook.extracting(event, extractor)
    subaudit.audit(event, 'a', 'b', 'c')
    with context_manager:
        pass
    extractor.assert_not_called()


def test_extracting_does_not_extract_before_enter(
    hook: Hook, event: str, extractor: Mock,
) -> None:
    context_manager = hook.extracting(event, extractor)
    subaudit.audit(event, 'a', 'b', 'c')
    with context_manager as extracts:
        pass
    assert extracts == []


def test_extracting_does_not_call_subscribe_before_enter(
    mocked_subscribe_unsubscribe_cls: type[Hook], event: str, extractor: Mock,
) -> None:
    subscribe = mocked_subscribe_unsubscribe_cls.subscribe
    hook = mocked_subscribe_unsubscribe_cls()
    hook.extracting(event, extractor)
    subscribe.assert_not_called()


def test_extracting_observes_between_enter_and_exit(
    hook: Hook, event: str, extractor: Mock,
) -> None:
    with hook.extracting(event, extractor):
        subaudit.audit(event, 'a', 'b', 'c')
    extractor.assert_called_once_with('a', 'b', 'c')


def test_extracting_extracts_between_enter_and_exit(
    hook: Hook, event: str, extractor: Mock,
) -> None:
    with hook.extracting(event, extractor) as extracts:
        subaudit.audit(event, 'a', 'b', 'c')
    assert extracts == [_Extract(args=('a', 'b', 'c'))]


def text_extracting_enter_calls_subscribe_exactly_once(
    mocked_subscribe_unsubscribe_cls: type[Hook], event: str, extractor: Mock,
) -> None:
    subscribe = mocked_subscribe_unsubscribe_cls.subscribe
    hook = mocked_subscribe_unsubscribe_cls()
    with hook.extracting(event, extractor):
        subscribe.assert_called_once()


def test_extracting_enter_passes_subscribe_same_event_and_hook(
    subtests: SubTests,
    mocked_subscribe_unsubscribe_cls: type[Hook],
    event: str,
    extractor: Mock,
) -> None:
    subscribe = mocked_subscribe_unsubscribe_cls.subscribe
    hook = mocked_subscribe_unsubscribe_cls()

    with hook.extracting(event, extractor):
        subscribe_hook, subscribe_event, _ = subscribe.calls[0].args
        with subtests.test(argument_name='hook'):
            assert subscribe_hook is hook
        with subtests.test(argument_name='event'):
            assert subscribe_event == event


def test_extracting_enter_passes_appender_to_subscribe(
    mocked_subscribe_unsubscribe_cls: type[Hook], event: str, extractor: Mock,
) -> None:
    subscribe = mocked_subscribe_unsubscribe_cls.subscribe
    hook = mocked_subscribe_unsubscribe_cls()

    with hook.extracting(event, extractor) as extracts:
        subscribe_call, = subscribe.calls
        _, _, subscribe_listener = subscribe_call.args

    subscribe_listener('a', 'b', 'c')
    assert extracts == [_Extract(args=('a', 'b', 'c'))]


def test_extracting_does_not_observe_after_exit(
    maybe_raise: Callable[[], None], hook: Hook, event: str, extractor: Mock,
) -> None:
    with contextlib.suppress(_FakeError):
        with hook.extracting(event, extractor):
            maybe_raise()
    subaudit.audit(event, 'a', 'b', 'c')
    extractor.assert_not_called()


def test_extracting_does_not_extract_after_exit(
    maybe_raise: Callable[[], None], hook: Hook, event: str, extractor: Mock,
) -> None:
    with contextlib.suppress(_FakeError):
        with hook.extracting(event, extractor) as extracts:
            maybe_raise()
    subaudit.audit(event, 'a', 'b', 'c')
    assert extracts == []


def test_extracting_exit_calls_unsubscribe_exactly_once(
    maybe_raise: Callable[[], None],
    mocked_subscribe_unsubscribe_cls: type[Hook],
    event: str,
    extractor: Mock,
) -> None:
    unsubscribe = mocked_subscribe_unsubscribe_cls.unsubscribe
    hook = mocked_subscribe_unsubscribe_cls()

    with contextlib.suppress(_FakeError):
        with hook.extracting(event, extractor):
            maybe_raise()

    unsubscribe.assert_called_once()


def test_extracting_exit_passes_unsubscribe_same_event_and_hook(
    subtests: SubTests,
    maybe_raise: Callable[[], None],
    mocked_subscribe_unsubscribe_cls: type[Hook],
    event: str,
    extractor: Mock,
) -> None:
    unsubscribe = mocked_subscribe_unsubscribe_cls.unsubscribe
    hook = mocked_subscribe_unsubscribe_cls()

    with contextlib.suppress(_FakeError):
        with hook.extracting(event, extractor):
            maybe_raise()

    unsubscribe_hook, unsubscribe_event, _ = unsubscribe.calls[0].args
    with subtests.test(argument_name='hook'):
        assert unsubscribe_hook is hook
    with subtests.test(argument_name='event'):
        assert unsubscribe_event == event


def test_extracting_exit_passes_appender_to_unsubscribe(
    maybe_raise: Callable[[], None],
    mocked_subscribe_unsubscribe_cls: type[Hook],
    event: str,
    extractor: Mock,
) -> None:
    unsubscribe = mocked_subscribe_unsubscribe_cls.unsubscribe
    hook = mocked_subscribe_unsubscribe_cls()

    with contextlib.suppress(_FakeError):
        with hook.extracting(event, extractor) as extracts:
            maybe_raise()

    unsubscribe_call, = unsubscribe.calls
    _, _, unsubscribe_listener = unsubscribe_call.args
    unsubscribe_listener('a', 'b', 'c')
    assert extracts == [_Extract(args=('a', 'b', 'c'))]


def test_extracting_observes_only_between_enter_and_exit(
    maybe_raise: Callable[[], None], hook: Hook, event: str, extractor: Mock,
) -> None:
    subaudit.audit(event, 'a')
    subaudit.audit(event, 'b', 'c')

    with contextlib.suppress(_FakeError):
        with hook.extracting(event, extractor):
            subaudit.audit('d')
            subaudit.audit('e', 'f')
            maybe_raise()

    subaudit.audit(event, 'g')
    subaudit.audit(event, 'h', 'i')

    assert extractor.mock_calls == [call('d'), call('e', 'f')]


def test_extracting_extracts_only_between_enter_and_exit(
    maybe_raise: Callable[[], None], hook: Hook, event: str, extractor: Mock,
) -> None:
    subaudit.audit(event, 'a')
    subaudit.audit(event, 'b', 'c')

    with contextlib.suppress(_FakeError):
        with hook.extracting(event, extractor) as extracts:
            subaudit.audit('d')
            subaudit.audit('e', 'f')
            maybe_raise()

    subaudit.audit(event, 'g')
    subaudit.audit(event, 'h', 'i')

    assert extracts == [_Extract(args=('d',)), _Extract(args=('e', 'f'))]


def test_extracting_subscribes_and_unsubscribes_same(
    maybe_raise: Callable[[], None],
    mocked_subscribe_unsubscribe_cls: type[Hook],
    event: str,
    extractor: Mock,
) -> None:
    subscribe = mocked_subscribe_unsubscribe_cls.subscribe
    unsubscribe = mocked_subscribe_unsubscribe_cls.unsubscribe
    hook = mocked_subscribe_unsubscribe_cls()

    with contextlib.suppress(_FakeError):
        with hook.extracting(event, extractor):
            maybe_raise()

    assert subscribe.mock_calls == unsubscribe.mock_calls


# FIXME: Retest some common cases with audit events from the standard library.


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
