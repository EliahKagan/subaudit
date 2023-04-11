"""Tests for the subaudit module."""

import sys
import unittest.mock
import uuid

import pytest

import subaudit


def _make_fake_event() -> str:
    """Create a randomly generated fake event name."""
    return f'test-subaudit-{uuid.uuid4()}'


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


def test_subscribed_listener_observes_event() -> None:
    event = _make_fake_event()
    listener = unittest.mock.Mock()
    hook = subaudit.Hook()
    hook.subscribe(event, listener)
    subaudit.audit(event, 'a', 'b', 'c')
    listener.assert_called_once_with('event', 'a', 'b', 'c')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
