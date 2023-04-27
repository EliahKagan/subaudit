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
Tests using events defined in the standard library (since Python 3.8).

These events are "real" in the sense that they represent realistic uses of the
``subaudit`` library, and also in the sense that tests outside this module use
events that are "fake" in the specific technical sense of being the kind of
test double known as a fake.

See https://docs.python.org/3/library/audit_events.html. We should be able to
listen to any event listed there, but these tests only try a select few.
"""

import io
import pathlib
import platform
import sys

import mock
from mock import call
import pytest

import subaudit
import tests.conftest as ct

_xfail_no_standard_audit_events_before_3_8 = pytest.mark.xfail(
    sys.version_info < (3, 8),
    reason='Python has no standard audit events before 3.8.',
    raises=AssertionError,
    strict=True,
)
"""
Mark expected failure by ``AssertionError`` due to no library events < 3.8.
"""


@_xfail_no_standard_audit_events_before_3_8
def test_can_listen_to_id_event(
    any_hook: ct.AnyHook, listener: ct.MockListener,
) -> None:
    """We can listen to the ``builtins.id`` event."""
    obj = object()
    obj_id = id(obj)
    with any_hook.listening('builtins.id', listener):
        id(obj)
    listener.assert_called_once_with(obj_id)


@_xfail_no_standard_audit_events_before_3_8
def test_can_listen_to_open_event(
    tmp_path: pathlib.Path, any_hook: ct.AnyHook, listener: ct.MockListener,
) -> None:
    """We can listen to the ``open`` event."""
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

    The ``sysaudit`` library backports this event to Python 3.7.
    """
    hook1, hook2 = make_hooks(2)
    listener1, listener2 = make_listeners(2)
    with hook1.listening('sys.addaudithook', listener1):
        with hook2.listening(event, listener2):
            listener1.assert_called_once_with()
