<!--
  Copyright (c) 2023 Eliah Kagan

  Permission to use, copy, modify, and/or distribute this software for any
  purpose with or without fee is hereby granted.

  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
  REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
  AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
  INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
  LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
  OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
  PERFORMANCE OF THIS SOFTWARE.
-->

# subaudit: Subscribe and unsubscribe for specific audit events

[Audit hooks](https://docs.python.org/3/library/audit_events.html) in Python
are called on all events, and they remain in place until the interpreter (or
subinterpreter) shuts down.

This library provides a higher-level interface that allows listeners to be
subscribed to specific audit events, and unsubscribed from them. It also
provides context managers for using that interface with a convenient notation
that ensures the listener is subscribed. The context managers are reentrant—you
can nest `with`-statements that listen to events. By default, a single audit
hook is used for any number of events and listeners.

The primary use case of this library is in writing test code.

## License

subaudit is licensed under [0BSD](https://spdx.org/licenses/0BSD.html), which
is a "public-domain equivalent" license. See
[**`LICENSE`**](https://github.com/EliahKagan/subaudit/blob/main/LICENSE).

## Usage

The best way to use subaudit is usually the `listening` context manager.

### The `subaudit.listening` context manager

```python
import subaudit

def listen_open(path, mode, flags):
    ...  # Handle the event.

with subaudit.listening('open', listen_open):
    ...  # Do something that may raise the event.
```

The listener—here, `listen_open`—is called with the event arguments each time
the event is raised. They are passed to the listener as separate positional
arguments (not as an `args` tuple).

In tests, it is convenient to use [`Mock`
objects](https://docs.python.org/3/library/unittest.mock.html#the-mock-class)
as listeners, since they record calls and provide a `mock_calls` attribute to
see them and various `assert_*` methods to make assertions about them:

```python
from unittest.mock import ANY, Mock
import subaudit

with subaudit.listening('open', Mock()) as listener:
    ...  # Do something that may raise the event.

listener.assert_any_call('/path/to/file.txt', 'r', ANY)
```

Note how, when the `listening` context manager is entered, it returns the
`listener` that was passed in, for convenience.

### The `subaudit.extracting` context manager

You may want to extract some information about calls to a list:

```python
from dataclasses import InitVar, dataclass
import subaudit

@dataclass(frozen=True)
class PathAndMode:  # See notebooks/open_event.ipynb about path and mode types.
    path: str
    mode: int
    flags: InitVar = None  # Opt not to record this argument.

with subaudit.extracting('open', PathAndMode) as extracts:
    ...  # Do something that may raise the event.

assert PathAndMode('/path/to/file.txt', 'r') in extracts
```

The extractor—here, `PathAndMode`—can be any callable that accepts the event
args as separate positional arguments. Entering the context manager returns an
initially empty list, which will be populated with *extracts* gleaned from the
event args. Each time the event is raised, the extractor is called and the
object it returns is appended to the list.

### `subaudit.subscribe` and `subaudit.unsubscribe`

Although you should usually use the `listening` or `extracting` context
managers instead, you can subscribe and unsubscribe listeners without a context
manager:

```python
import subaudit

def listen_open(path, mode, flags):
    ...  # Handle the event.

subaudit.subscribe('open' listen_open)
try:
    ...  # Do something that may raise the event.
finally:
    subaudit.unsubscribe('open', listen_open)
```

### `Hook` objects
