"""subaudit: Subscribe and unsubscribe for specific audit events."""

__all__ = [
    'ContextManagerFactory',
    'audit',
    'addaudithook',
    'Hook',
    'shared',  # A global Hook instance.
    'subscribe',  # Calls shared.subscribe.
    'unsubscribe',  # Calls shared.unsubscribe.
    'listening',  # Calls shared.listening.
    'extracting',  # Calls shared.extracting
    'skip_if_unavailable',
]  # FIXME: Delete the above comments. Move any key info to items' docstrings.

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


# FIXME: Probably move some of this class's docstring to the module docstring.
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
    number of listeners. This is because a Hook stores store each event's
    listeners in an immutable sequence, rebuilt each time a listener is
    subscribed or unsubscribed. (This is part of how consistent state is
    maintained at all times, so the installed audit hook doesn't need to
    synchronize with subscribe and unsubscribe.) Subscribing N listeners to the
    same event, without unsubscribing any, takes O(N**2) time. If you need to
    have more than a couple hundred listeners on the same event at the same
    time, especially if you are also frequently subscribing and unsubscribing
    listeners to that same event, this may not be the right tool for the job.
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

    # FIXME: Add a __repr__ that gives *some* useful debugging information.
    #        Consider the tradeoff that walking the whole table will require
    #        that we lock, but this could lead to deadlocks while debugging the
    #        methods the lock is really for (subscribe and unsubscribe).

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

    @contextmanager
    def listening(
        self, event: str, listener: Callable[..., None],
    ) -> Generator[None, None, None]:
        """Context manager to subscribe and unsubscribe an event listener."""
        self.subscribe(event, listener)
        try:
            yield
        finally:
            self.unsubscribe(event, listener)

    @contextmanager
    def extracting(
        self, event: str, extractor: Callable[..., _R],
    ) -> Generator[List[_R], None, None]:
        """Context manager to provide a list of custom-extracted event data."""
        extracts: List[_R] = []

        def append_extract(*args: Any) -> None:
            extracts.append(extractor(*args))

        with self.listening(event, append_extract):
            yield extracts

    @staticmethod
    def _fail_unsubscribe(
        event: str, listener: Callable[..., None],
    ) -> NoReturn:
        """Raise an error for an unsuccessful attempt to detach a listener."""
        raise ValueError(f'{event!r} listener {listener!r} never subscribed')

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


# FIXME: Add stuff for the global Hook instance.


# FIXME: Add skip_if_unavailable.
