"""Tests for the subaudit module."""

import sys
from unittest.mock import Mock
import uuid

import pytest

import subaudit
from subaudit import Hook


@pytest.fixture
def event() -> str:
    """Create a randomly generated fake event name."""
    return f'test-subaudit-{uuid.uuid4()}'


@pytest.fixture
def listener() -> Mock:
    """Create a mock listener."""
    return Mock()


@pytest.fixture
def hook() -> Hook:
    """Create a Hook instance."""
    return Hook()


# FIXME: Change each skipif to xfail with a condition.
#        Use a raises argument where appropriate.


@pytest.mark.skipif(sys.version_info < (3, 8),
                    reason="Python 3.8+ has sys.audit")
def test_audit_is_sys_audit_since_3_8() -> None:
    assert subaudit.audit is sys.audit


@pytest.mark.skipif(sys.version_info >= (3, 8),
                    reason="Python 3.8+ has sys.audit")
def test_audit_is_sysaudit_audit_before_3_8() -> None:
    import sysaudit
    assert subaudit.audit is sysaudit.audit


@pytest.mark.skipif(sys.version_info < (3, 8),
                    reason="Python 3.8+ has sys.addaudithook")
def test_addaudithook_is_sys_addaudithook_since_3_8() -> None:
    assert subaudit.addaudithook is sys.addaudithook


@pytest.mark.skipif(sys.version_info >= (3, 8),
                    reason="Python 3.8+ has sys.addaudithook")
def test_addaudithook_is_sysaudit_addaudithook_before_3_8() -> None:
    import sysaudit
    assert subaudit.addaudithook is sysaudit.addaudithook


def test_subscribed_listener_observes_event(
        event: str, listener: Mock, hook: Hook) -> None:
    hook.subscribe(event, listener)
    subaudit.audit(event, 'a', 'b', 'c')
    listener.assert_called_once_with('event', 'a', 'b', 'c')


def test_unsubscribed_listener_does_not_observe_event(
        event: str, listener: Mock, hook: Hook) -> None:
    hook.subscribe(event, listener)
    hook.unsubscribe(event, listener)
    subaudit.audit(event, 'a', 'b', 'c')
    listener.assert_not_called()


def test_cannot_unsubscribe_if_never_subscribed(
        event: str, listener: Mock, hook: Hook) -> None:
    with pytest.raises(ValueError):
        hook.unsubscribe(event, listener)


def test_cannot_unsubscribe_if_no_longer_subscribed(
        event: str, listener: Mock, hook: Hook) -> None:
    hook.subscribe(event, listener)
    hook.unsubscribe(event, listener)
    with pytest.raises(ValueError):
        hook.unsubscribe(event, listener)


@pytest.mark.parametrize('count', [0, 2, 3, 10])
def test_listener_observes_event_as_many_times_as_subscribed(
        count: int, event: str, listener: Mock, hook: Hook) -> None:
    for _ in range(count):
        hook.subscribe(event, listener)
    subaudit.audit(event)
    assert listener.call_count == count


@pytest.mark.parametrize('count', [2, 3, 10])
def test_can_unsubscribe_as_many_times_as_subscribed(
        count:int, event: str, listener: Mock, hook: Hook) -> None:
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
        count: int, event: str, listener: Mock, hook: Hook) -> None:
    for _ in range(count):
        hook.subscribe(event, listener)
    with pytest.raises(ValueError):
        for _ in range(count + 1):
            hook.subscribe(event, listener)


# FIXME: Test that unsubscribing removes the last-subscribed equal listener.


# FIXME: Test subscribe and unsubscribe with multiple (unequal) listeners.


# FIXME: Test the context managers from Hook.listening and Hook.extracting.


# FIXME: Test that a listener cannot be unsubscribed from a different Hook.


# FIXME: Test that a Hook's first subscription adds an audit hook.


# FIXME: Test that a Hook's subsequent subscriptions don't add audit hooks.


# FIXME: Test that a second Hook's first subscription adds an audit hook.


# FIXME: Retest some common cases with audit events from the standard library.


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
