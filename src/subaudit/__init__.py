"""subaudit: Subscribe and unsubscribe for specific audit events."""

__all__ = [
    'audit',
    'addaudithook',
    'Hook',
    'SyncHook',
    'shared',  # A global SyncHook instance.
    'subscribe',  # Calls shared.subscribe.
    'unsubscribe',  # Calls shared.unsubscribe.
    'listening',  # Calls shared.listening.
    'extracting',  # Calls shared.extracting
    'skip_if_unavailable',
]  # FIXME: Delete the above comments. Move any key info to items' docstrings.

import contextlib
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
)

try:
    from sys import audit, addaudithook
except ImportError:
    from sysaudit import audit, addaudithook

_R = TypeVar('_R')
"""Type variable used to represent the return type of an extractor."""

_table: Optional[MutableMapping[str, Tuple[Callable[..., None], ...]]] = None
"""Table mapping each event to its listeners, or None if not yet needed."""


def _hook(event: str, args: tuple[Any, ...]) -> None:
    """Single audit hook used for all events and handlers."""
    try:
        # Subscripting a dict with str keys should be sufficiently protected by
        # the GIL in CPython. This doesn't protect the table rows. But those
        # are tuples that we always replace, rather than lists that we mutate,
        # so we should observe consistent state.
        listeners = _table[event]
    except KeyError:
        return

    for listener in listeners:
        listener(*args)


def _subscribe(event: str, listener: Callable[..., None]) -> None:
    global _table

    if _table is None:
        _table = {}
        addaudithook(_hook)

    old_listeners = _table.get(event, ())
    _table[event] = (*old_listeners, listener)


def _fail_unsubscribe(event: str, listener: Callable[..., None]) -> NoReturn:
    """Raise an exception for an unsuccessful attempt to detach a listener."""
    raise ValueError(f'{event!r} listener {listener!r} never subscribed')


def _unsubscribe(event: str, listener: Callable[..., None]) -> None:
    if _table is None:
        _fail_unsubscribe(event, listener)

    try:
        listeners = _table[event]
    except KeyError:
        _fail_unsubscribe(event, listener)

    # Work with the sequence in reverse to remove the most recent listener.
    listeners_reversed = list(reversed(listeners))
    try:
        listeners_reversed.remove(listener)
    except ValueError:
        _fail_unsubscribe(event, listener)

    if listeners_reversed:
        _table[event] = tuple(reversed(listeners_reversed))
    else:
        del _table[event]


@contextlib.contextmanager
def _listening(
    event: str,
    listener: Callable[..., None],
) -> Generator[None, None, None]:
    _subscribe(event, listener)
    try:
        yield
    finally:
        _unsubscribe(event, listener)


@contextlib.contextmanager
def _extracting(
    event: str,
    extractor: Callable[..., _R],
) -> Generator[List[_R], None, None]:
    extracts = []
    with _listening(event, lambda *args: extracts.append(extractor(*args))):
        yield extracts


# FIXME: Move the code each method uses into the method and have it use state
#        belonging to the Hook instance rather than a global _table, removing
#        the old code, so nearly all state and behavior of Hook objects is
#        coded in this class. This must be done so multiple Hook instances are
#        independent. It is the very minimum needed to get this class to a
#        state that may be worth using, and to satisfy the tests' requirements.
#
# FIXME: Rework "handles writing a single reference as an atomic operation",
#        which disregards the other issue of the dictionary lookup.
#
# FIXME: Move as much of this class's docstring as is reasonable to the module
#        docstring, but wait to do so until the SyncHook subclass is written.
#
# FIXME: Note, somewhere, that a future version of the library *might* scale up
#        to an asymptotically faster data structure for rows containing large
#        numbers (thousands? more?) of listeners. OR, explain why it won't.
#
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

    The subscribe and unsubscribe methods are NOT thread-safe. However, so long
    as the caller ensures no data race happens between calls to either or both
    of those methods, the state of the Hook should not be corrupted... IF the
    Python interpreter is CPython or otherwise handles writing a single
    reference as an atomic operation. This is to say: at least on CPython,
    segfaults or very strange behavior shouldn't happen due to an event firing,
    even if a listener is subscribing or unsubscribing at the same time.

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

    __slots__ = ()

    def subscribe(self, event: str, listener: Callable[..., None]) -> None:
        """Attach a detachable listener to an event."""
        _subscribe(event, listener)

    def unsubscribe(self, event: str, listener: Callable[..., None]) -> None:
        """Detach a listener that was attached to an event."""
        _unsubscribe(event, listener)

    def listening(
        self,
        event: str,
        listener: Callable[..., None],
    ) -> ContextManager[None]:
        """Context manager to subscribe and unsubscribe an event listener."""
        return _listening(event, listener)

    def extracting(
        self,
        event: str,
        extractor: Callable[..., _R],
    ) -> ContextManager[List[_R]]:
        """Context manager to provide a list of custom-extracted event data."""
        return _extracting(event, extractor)


# FIXME: Add SyncHook and the top-level stuff for the global SyncHook instance.


# FIXME: Add skip_if_unavailable.
