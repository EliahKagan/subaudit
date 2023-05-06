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
subaudit: Subscribe and unsubscribe for specific audit events.

This library provides a higher-level interface that allows listeners to be
subscribed to specific audit events, and unsubscribed from them. It provides
context managers for using that interface with a convenient notation that
ensures the listener is unsubscribed. The context managers are reentrant — you
can nest ``with``-statements that listen to events. By default, a single audit
hook is used for any number of events and listeners.

The primary use case for this library is in writing test code. Although it can
be used and customized in a few ways, the top-level ``subaudit.listening`` and
``subaudit.extracting`` context manager functions are usually sufficient.

Example usage:

    import subaudit

    def listen_open(path, mode, flags):
        ...  # Handle the event.

    with subaudit.listening('open', listen_open):
        ...  # Do something that may raise the event.

See the project ``README.md`` for more information on usage.
"""

__all__ = [
    'LockContextManagerFactory',
    'audit',
    'addaudithook',
    'Hook',
    'subscribe',
    'unsubscribe',
    'listening',
    'extracting',
    'skip_if_unavailable',
]

import contextlib
import sys
import threading
from typing import (
    Any,
    Callable,
    ContextManager,
    Generator,
    List,
    MutableMapping,
    NoReturn,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
import unittest

if sys.version_info < (3, 8):
    from sysaudit import addaudithook, audit
else:
    from sys import addaudithook, audit

_F = TypeVar('_F', bound=Callable[..., None])
"""Type variable used to represent the callable type of an event listener."""

_R = TypeVar('_R')
"""Type variable used to represent the return type of an extractor."""

LockContextManager = Union[ContextManager[None], ContextManager[bool]]
"""
Type alias covering ``None``-returning and ``bool``-returning context managers.

``Lock``, ``RLock``, and other synchronization primitives' ``__enter__``
methods return ``bool`` (because ``acquire`` does). Most context managers that
may be used in their place, including ``contextlib.nullcontext``, return
``None``.

This alias differs subtly from ``Optional[bool]``. With ``Optional[bool]``, the
same object could return a ``bool`` on some calls and ``None`` on others.
"""

LockContextManagerFactory = Callable[[], LockContextManager]
"""
Type alias for classes or factory functions returning a ``LockContextManager``.
"""


class Hook:
    """
    Audit hook wrapper. Subscribes and unsubscribes specific-event listeners.

    Listeners are subscribed and unsubscribed for specific auditing events.
    Only one audit hook (per ``Hook`` instance) is used. It is installed the
    first time a listener is subscribed to any event via the ``Hook`` instance;
    if the instance is never used, no audit hook is installed. A program rarely
    needs multiple ``Hook`` instances, even with many listeners and events. The
    ``subaudit`` module provides top-level ``subscribe``, ``unsubscribe``,
    ``listening``, and ``extracting`` functions, which use a pre-created
    ``Hook`` instance.

    The ``subscribe`` and ``unsubscribe`` methods, but not the installed audit
    hook, are by default protected by a mutex. The audit hook can be called at
    any time, including as ``subscribe`` or ``unsubscribe`` runs: it is called
    on all audit events, filtering for those of interest. However, if the
    Python interpreter is CPython — or another implementation where writing an
    attribute reference is atomic, and writing or deleting an item in a dict
    with string keys is atomic — then the ``Hook``'s state should not be
    corrupted. In short, on CPython, strange behavior and segfaults shouldn't
    happen due to an event firing, even if a listener subscribes or
    unsubscribes at the same time.

    ``Hook`` objects are not optimized for the case of an event having a large
    number of listeners. This is because a ``Hook`` stores each event's
    listeners in an immutable sequence, rebuilt each time a listener is
    subscribed or unsubscribed. (This is part of how consistent state is
    maintained, so the audit hook doesn't need to synchronize with subscribe
    and unsubscribe.) Subscribing N listeners to the same event without
    unsubscribing takes O(N²) time. If you need more than a couple hundred
    listeners on the same event at the same time, especially if you also
    frequently subscribe and unsubscribe listeners to that same event, this may
    be the wrong tool.
    """

    __slots__ = ('_lock', '_hook_installed', '_table')

    _lock: LockContextManager
    """
    Mutex or other context manager. Protects ``subscribe`` and ``unsubscribe``.
    """

    _hook_installed: bool
    """Whether the audit hook is installed yet."""

    _table: MutableMapping[str, Tuple[Callable[..., None], ...]]
    """Table that maps each event to its listeners."""

    def __init__(
        self, *, sub_lock_factory: Optional[LockContextManagerFactory] = None,
    ) -> None:
        """
        Make an audit hook wrapper, which will use its own audit hook.

        :param sub_lock_factory: If passed, this is called instead of
            ``threading.Lock`` to create a context manager object, which is
            used as a mutex during subscribing and unsubscribing. To forgo
            locking, pass ``contextlib.nullcontext``.
        """
        if sub_lock_factory is None:
            sub_lock_factory = threading.Lock
        self._lock = sub_lock_factory()
        self._hook_installed = False
        self._table = {}

    def __repr__(self) -> str:
        """Representation for debugging. Not runnable as Python code."""
        return f'<{type(self).__name__} at {id(self):#x}: {self._summarize()}>'

    def subscribe(self, event: str, listener: Callable[..., None]) -> None:
        """
        Attach a detachable listener to an event.

        :param event: The name of the event to subscribe the listener to.
        :param listener: The callable listener to be subscribed to the event.

        Overriding ``subscribe`` and ``unsubscribe`` will customize the
        behavior of ``listening`` and ``extracting``.
        """
        # Produce a suitable key for the event, raising TypeError if we can't.
        if type(event) is not str:
            if not isinstance(event, str):
                raise TypeError(f"event must be 'str', not {type(event)!r}")
            event = f'{event!s}'  # Exactly str, even when str(event) isn't.

        with self._lock:
            if not self._hook_installed:
                addaudithook(self._hook)
                self._hook_installed = True

            old_listeners = self._table.get(event, ())
            self._table[event] = (*old_listeners, listener)

    def unsubscribe(self, event: str, listener: Callable[..., None]) -> None:
        """
        Detach a listener that was attached to an event.

        :param event: The name of the event to unsubscribe the listener from.
        :param listener: The callable listener to be unsubscribed from the
            event.

        Overriding ``subscribe`` and ``unsubscribe`` will customize the
        behavior of ``listening`` and ``extracting``.
        """
        with self._lock:
            try:
                listeners = self._table[event]
            except KeyError:
                self._fail_unsubscribe(event, listener)

            # Search in reverse, to find the most recent matching listener.
            try:
                reverse_last_index = listeners[::-1].index(listener)
            except ValueError:
                self._fail_unsubscribe(event, listener)

            # Remove it. (Use positive indexing to avoid slicing from "-0".)
            last_index = len(listeners) - 1 - reverse_last_index
            listeners = listeners[:last_index] + listeners[(last_index + 1):]

            # Replace the row, unless empty, in which case remove it.
            if listeners:
                self._table[event] = listeners
            else:
                del self._table[event]

    def listening(self, event: str, listener: _F) -> ContextManager[_F]:
        """
        Context manager to subscribe and unsubscribe an event listener.

        :param event: The name of the event to subscribe the listener to.
        :param listener: The callable listener to be subscribed to the event.
        :return: A context manager that subscribes the listener when entered
            and unsubscribes the listener when exited.
        """
        return self._make_listening(event, listener)

    def extracting(
        self, event: str, extractor: Callable[..., _R],
    ) -> ContextManager[List[_R]]:
        """
        Context manager to provide a list of custom-extracted event data.

        This function returns a context manager object. Entering the context
        manager returns an initially empty list. The list is populated with
        extracts produced by calling ``extractor`` with the event args. Each
        time ``event`` occurs, an extract is appended to the list, until the
        context manager is exited.

        :param event: The name of the event to extract data from.
        :param extractor: The callable extractor that selects data from event
            args.
        :return: A context manager that starts extracting data from the event
            when entered, and stops when exited.
        """
        return self._make_extracting(event, extractor)

    def _hook(self, event: str, args: Tuple[Any, ...]) -> None:
        """Single audit hook used for all events and handlers."""
        try:
            # Subscripting a dict with str keys should be sufficiently
            # protected by the GIL in CPython. This doesn't protect the table
            # rows. But those are tuples that we always replace, rather than
            # lists that we mutate, so we should observe consistent state.
            listeners = self._table[event]
        except KeyError:
            return

        for listener in listeners:
            listener(*args)

    def _summarize(self) -> str:
        """
        Summarize the state of the ``Hook`` instance. Used as part of the repr.

        For now, just include info CPython lets us get safely without a lock.
        """
        if not self._hook_installed:
            return 'audit hook not installed'

        num_events = len(self._table)
        if num_events == 1:
            return f'watching {num_events} event'
        return f'watching {num_events} events'

    @staticmethod
    def _fail_unsubscribe(
        event: str, listener: Callable[..., None],
    ) -> NoReturn:
        """Raise an error for an unsuccessful attempt to detach a listener."""
        raise ValueError(f'{event!r} listener {listener!r} never subscribed')

    @contextlib.contextmanager
    def _make_listening(
        self, event: str, listener: _F,
    ) -> Generator[_F, None, None]:
        """
        Helper for ``listening``.

        Callers shouldn't assume ``listening`` returns
        ``GeneratorContextManager``. This helper allow ``listening`` to have
        the desired type annotations. Subclasses may override ``listening`` but
        shouldn't override or call this.
        """
        self.subscribe(event, listener)
        try:
            yield listener
        finally:
            self.unsubscribe(event, listener)

    @contextlib.contextmanager
    def _make_extracting(
        self, event: str, extractor: Callable[..., _R],
    ) -> Generator[List[_R], None, None]:
        """
        Helper for ``extracting``.

        Callers shouldn't assume ``extracting`` returns
        ``GeneratorContextManager``. This helper allows ``extracting`` to have
        the desired type annotations. Subclasses may override ``extracting``
        but shouldn't override or call this.
        """
        extracts: List[_R] = []

        def append_extract(*args: Any) -> None:
            extracts.append(extractor(*args))

        with self.listening(event, append_extract):
            yield extracts


_global_instance = Hook()
"""
``Hook`` instance used by the top-level functions.

The module-level ``subscribe``, ``unsubscribe``, ``listening``, and
``extracting`` functions use this instance. This should not be confused with
the behavior of each ``Hook`` object in installing (at most) one actual
auditing event hook.
"""

subscribe = _global_instance.subscribe
unsubscribe = _global_instance.unsubscribe
listening = _global_instance.listening
extracting = _global_instance.extracting


skip_if_unavailable = unittest.skipIf(
    sys.version_info < (3, 8),
    'Python Runtime Audit Hooks (PEP 578) were introduced in Python 3.8.',
)
"""
Skip a ``unittest`` test if the standard library doesn't have audit events.
"""
