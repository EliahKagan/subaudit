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
are called on all events, and they remain in place until the interpreter shuts
down.

This library provides a higher-level interface that allows listeners to be
subscribed to specific audit events, and unsubscribed from them. It also
provides context managers for using that interface with a convenient notation
that ensures the listener is unsubscribed. The context managers are
reentrant—you can nest `with`-statements that listen to events. By default, a
single audit hook is used for any number of events and listeners.

The primary use case for this library is in writing test code.

## License

subaudit is licensed under [0BSD](https://spdx.org/licenses/0BSD.html), which
is a ["public-domain
equivalent"](https://en.wikipedia.org/wiki/Public-domain-equivalent_license)
license. See
[**`LICENSE`**](https://github.com/EliahKagan/subaudit/blob/main/LICENSE).

## Compatibility

The subaudit library can be used to observe [audit events generated by the
Python interpreter and standard
library](https://docs.python.org/3/library/audit_events.html), as well as
custom audit events. It requires Python 3.7 or later. It is most useful on
Python 3.8 or later, because [audit events were introduced in Python
3.8](https://peps.python.org/pep-0578/). On Python 3.7, subaudit uses [the
*sysaudit* library](https://pypi.org/project/sysaudit/) to support audit
events, but the Python interpreter and standard library still do not provide
any events, so only custom events can be used on Python 3.7.

To avoid the performance cost of explicit locking in the audit hook, [some
operations are assumed atomic](#Locking). I believe these assumptions are
correct for CPython, as well as PyPy and some other implementations, but there
may exist Python implementations on which these assumptions don't hold.

## Basic usage

### The `listening` context manager

The best way to use subaudit is usually the `subaudit.listening` context
manager.

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

### The `extracting` context manager

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

### `subscribe` and `unsubscribe`

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

Attempting to unsubscribe a listener that is not subscribed raises
`ValueError`. Currently, subaudit provides no feature to make this succeed
silently instead. But you can suppress the exception:

```python
with contextlib.suppress(ValueError):
    subaudit.unsubscribe('glob.glob', possibly_subscribed_listener)
```

## Nesting

To unsubscribe a listener from an event, it must be subscribed to the event.
Subject to this restriction, calls to `subscribe` and `unsubscribe` can happen
in any order and use of `listening` and `extracting` may be arbitrarily nested.

`listening` and `extracting` support reentrant use with both the same event and
different events. Here's an example with three `listening` contexts:

```python
from unittest.mock import Mock, call

listen_to = Mock()  # Let us assert calls to child mocks in a specific order.

with subaudit.listening('open', print):  # Print all open events' arguments.
    with subaudit.listening('open', listen_to.open):  # Log opening.
        with subaudit.listening('glob.glob', listen_to.glob):  # Log globbing.
            ...  # Do something that may raise the events.

assert parent.mock_calls == ...  # Assert a specific order of calls.
```

(That is written out to make the nesting clear. You can, as always, use a
single `with`-statement with commas instead.)

Here's an example with both `listening` and `extracting` contexts.

```python
from unittest.mock import Mock, call

def extract(*args):
    return args

with (
    subaudit.extracting('pathlib.Path.glob', extract) as glob_extracts,
    subaudit.listening('pathlib.Path.glob', Mock()) as glob_listener,
    subaudit.extracting('pathlib.Path.rglob', extract) as rglob_extracts,
    subaudit.listening('pathlib.Path.rglob', Mock()) as rglob_listener,
):
    ...  # Do something that may raise the events.

# Assert something about, or otherwise use, the mocks glob_listener and
# rglob_listener, as well as the lists glob_extracts and rglob_extracts.
...
```

(That example uses [parenthesized context
managers](https://docs.python.org/3/whatsnew/3.10.html#parenthesized-context-managers),
which were introduced in Python 3.10.)

## Specialized usage

### `Hook` objects

Each instance of the `subaudit.Hook` class represents a single audit hook that
supports subscribing and unsubscribing listeners for any number of events, with
methods corresponding to the four top-level functions listed above. Separate
`Hook` instances use separate audit hooks. The `Hook` class exists for three
purposes:

- It supplies the behavior of the top-level `listening`, `extracting`,
  `subscribe`, and `unsubscribe` functions, which correspond to the same-named
  methods on a global `Hook` instance.
- It allows multiple audit hooks to be used, for special cases where that might
  be desired.
- It facilitates customization, as detailed below.

The actual audit hook that a `Hook` object encapsulates is not installed until
the first listener is subscribed. This happens on the first call to the
object's `subscribe` method, or the first time a context manager object
obtained by calling its `listening` or `extracting` method is entered. This is
also true of the global `Hook` instance used by the top-level functions—merely
importing `subaudit` does not install an audit hook.

Whether the top-level functions are bound methods of a `Hook` instance, or
delegate in some other way to those methods on an instance, is currently
considered an implementation detail.

### Deriving from `Hook`

You can derive from `Hook` to provide custom behavior for subscribing and
unsubscribing, by overriding the `subscribe` and `unsubscribe` methods. You can
also override the `listening` and `extracting` methods, though that may be less
useful. Overridden `subscribe` and `unsubscribe` methods are automatically used
by `listening` and `extracting`.

Whether `extracting` uses `listening`, or directly calls `subscribe` and
`unsubscribe`, is currently considered an implementation detail.

### Locking

Consider two possible cases of race conditions:

#### 1. Between a the audit hook and `subscribe` or `unsubscribe` (no locking)

In this scenario, a `Hook` object's installed audit hook runs at the same time
as a listener is subscribed or unsubscribed.

This is likely to occur often and it cannot be prevented, because audit hooks
are called for all audit events. For the same reason, locking in the audit hook
has performance implications. Instead of having audit hooks take locks,
subaudit relies on each of these operations being atomic:

- Writing an attribute reference, when it is a simple write to an instance
  dictionary or a slot. Writing an attribute need not be atomic when, for
  example, `__setattr__` has been overridden.
- Writing or deleting a ``str`` key in a dictionary whose keys are all ``str``.
  Note that the search need not be atomic, but the dictionary must always be
  observed to be in a valid state.

#### 2. Between calls to `subscribe` and/or `unsubscribe` (locking by default)

In this scenario, two listeners are subscribed at a time, or unsubscribed at a
time, or one listener is subscribed while another (or the same) listener is
unsubscribed.

This is less likely to occur and much easier to avoid. But it is also harder to
make safe without a lock. Subscribing and unsubscribing are unlikely to happen
occur at a *sustained* high rate, so locking is unlikely to be a performance
bottleneck. So, *by default*, subscribing and unsubscribing are synchronized
with a `threading.Lock`, to ensure that shared state is not corrupted.

You should not usually change this. But if you want to, you can construct a
`Hook` object by calling `Hook(sub_lock_factory=...)` instead of `Hook`, where
`...` is a type or other context manager factory to be used instead of
`threading.Lock`. In particular, to disable locking, pass
`contextlib.nullcontext`.

## Functions related to compatibility

### `subaudit.addaudithook` and `subaudit.audit`

<!-- FIXME: Write this subsection. -->

### `@skip_if_unavailable`

<!-- FIXME: Write this subsection. -->

## Acknowledgements

<!-- Write this section. -->
