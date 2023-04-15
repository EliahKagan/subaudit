"""subaudit: Subscribe and unsubscribe for specific audit events."""

__all__ = [
    'ContextManagerFactory',
    'audit',
    'addaudithook',
    'Hook',
    'subscribe',
    'unsubscribe',
    'listening',
    'extracting',
    'skip_if_unavailable',
]

from contextlib import AbstractContextManager, contextmanager
import sys
import threading
from typing import (
    Any,
    Callable,
    Generator,
    List,
    MutableMapping,
    NoReturn,
    Optional,
    Tuple,
    TypeVar,
)

if sys.version_info < (3, 8):
    from sysaudit import audit, addaudithook
else:
    from sys import audit, addaudithook

_R = TypeVar('_R')
"""Type variable used to represent the return type of an extractor."""

ContextManagerFactory = Callable[[], AbstractContextManager]
"""Type alias for classes or factory functions returning context managers."""


class Hook:
    """
    Audit hook wrapper. Subscribes and unsubscribes specific-event listeners.

    Listeners subscribe to specific auditing events and may unsubscribe from
    them. Only one audit hook is actually installed (per Hook instance). The
    actual audit hook for a Hook instance is installed the first time a
    listener subscribes via the Hook instance, so if the Hook is never needed,
    no audit hook is installed. The suggested approach is to use only a small
    number of Hook instances, often just one, even if many different listeners
    will be subscribed and unsubscribed for many (or few) different events.

    The subscribe and unsubscribe methods, but not the installed audit hook,
    are protected by a mutex. The hook can be called at any time, including as
    subscribe or unsubscribe runs, because it is called on all audit events
    (and it filters out all but those of interest). However, IF the Python
    interpreter is CPython (or another implementation that handles writing an
    attribute reference, or writing/deleting a dict item with a string key,
    atomically), the state of the Hook shouldn't be corrupted. At least on
    CPython, strange behavior and segfaults shouldn't happen from an event
    firing, even if a listener subscribes or unsubscribes at the same time.

    Hook objects are not optimized for the case of an event having a large
    number of listeners. This is because a Hook stores each event's listeners
    in an immutable sequence, rebuilt each time a listener is subscribed or
    unsubscribed. (This is part of how consistent state is maintained, so the
    audit hook doesn't need to synchronize with subscribe and unsubscribe.)
    Subscribing N listeners to the same event without unsubscribing takes
    O(N**2) time. If you need more than a couple hundred listeners on the same
    event at the same time, especially if you also frequently subscribe and
    unsubscribe listeners to that same event, this may be the wrong tool.
    """

    __slots__ = ('_lock', '_hook_installed', '_table')

    _lock: AbstractContextManager
    """Mutex or other context manager used to protect subscribe/unsubscribe."""

    _hook_installed: bool
    """Whether the audit hook is installed yet."""

    _table: MutableMapping[str, Tuple[Callable[..., None], ...]]
    """Table that maps each event to its listeners."""

    def __init__(
        self, *, sub_lock_factory: Optional[ContextManagerFactory] = None,
    ) -> None:
        """
        Make an audit hook wrapper, which will use its own audit hook.

        If sub_lock_factory is passed, it is called and the result must be a
        context manager object, which is used as a mutex during subscribing
        and unsubscribing. To forgo locking, pass contextlib.nullcontext.
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
        """Attach a detachable listener to an event."""
        with self._lock:
            if not self._hook_installed:
                addaudithook(self._hook)
                self._hook_installed = True

            old_listeners = self._table.get(event, ())
            self._table[event] = (*old_listeners, listener)

    def unsubscribe(self, event: str, listener: Callable[..., None]) -> None:
        """Detach a listener that was attached to an event."""
        with self._lock:
            try:
                listeners = self._table[event]
            except KeyError:
                self._fail_unsubscribe(event, listener)

            # We search in reverse, to remove the latest matching listener.
            listeners_reversed = list(reversed(listeners))
            try:
                listeners_reversed.remove(listener)
            except ValueError:
                self._fail_unsubscribe(event, listener)

            if listeners_reversed:
                self._table[event] = tuple(reversed(listeners_reversed))
            else:
                del self._table[event]

    def listening(
        self, event: str, listener: Callable[..., None],
    ) -> AbstractContextManager[None]:
        """Context manager to subscribe and unsubscribe an event listener."""
        return self._make_listening(event, listener)

    def extracting(
        self, event: str, extractor: Callable[..., _R],
    ) -> AbstractContextManager[List[_R]]:
        """Context manager to provide a list of custom-extracted event data."""
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
        Summarize the state of the Hook instance. Used as part of the repr.

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

    @contextmanager
    def _make_listening(
        self, event: str, listener: Callable[..., None],
    ) -> Generator[None, None, None]:
        """
        Helper for listening.

        Callers shouldn't assume listening returns GeneratorContextManager.
        This helper allow listening to have the desired type annotations.
        Subclasses may override listening but shouldn't override or call this.
        """
        self.subscribe(event, listener)
        try:
            yield
        finally:
            self.unsubscribe(event, listener)

    @contextmanager
    def _make_extracting(
        self, event: str, extractor: Callable[..., _R],
    ) -> Generator[List[_R], None, None]:
        """
        Helper for extracting.

        Callers shouldn't assume extracting returns GeneratorContextManager.
        This helper allows extracting to have the desired type annotations.
        Subclasses may override extracting but shouldn't override or call this.
        """
        extracts: List[_R] = []

        def append_extract(*args: Any) -> None:
            extracts.append(extractor(*args))

        with self._make_listening(event, append_extract):
            yield extracts


_global_instance = Hook()
"""
Hook instance used by the top-level functions.

The module-level subscribe, unsubscribe, listening, and extracting functions
use this instance. This should not be confused with the behavior of each Hook
object in installing (at most) one actual auditing event hook.
"""

subscribe = _global_instance.subscribe
unsubscribe = _global_instance.unsubscribe
listening = _global_instance.listening
extracting = _global_instance.extracting


# FIXME: Add skip_if_unavailable.
