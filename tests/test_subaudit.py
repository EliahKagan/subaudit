"""Tests for the subaudit module."""

import sys

import pytest

import subaudit


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Python 3.8+ has sys.audit")
def test_audit_is_sys_audit_since_3_8() -> None:
    assert subaudit.audit is sys.audit


@pytest.mark.skipif(sys.version_info >= (3, 8), reason="Python 3.8+ has sys.audit")
def test_audit_is_sysaudit_audit_before_3_8() -> None:
    import sysaudit
    assert subaudit.audit is sysaudit.audit


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Python 3.8+ has sys.addaudithook")
def test_addaudithook_is_sys_addaudithook_since_3_8() -> None:
    assert subaudit.addaudithook is sys.addaudithook


@pytest.mark.skipif(sys.version_info >= (3, 8), reason="Python 3.8+ has sys.addaudithook")
def test_addaudithook_is_sysaudit_addaudithook_before_3_8() -> None:
    import sysaudit
    assert subaudit.addaudithook is sysaudit.addaudithook


def test_subscribed_listener_observes_event():
    raise NotImplementedError  # FIXME: Write this test.


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
