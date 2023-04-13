"""subaudit: Subscribe and unsubscribe for specific audit events."""

__all__ = [
    'audit',
    'addaudithook',
    'Hook',  # FIXME: Methods: subscribe, unsubscribe, listening, extracting.
    'skip_if_unavailable',
]

import contextlib
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
    """Attach a detachable listener to an event."""
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
    """Detach a listener that was attached to an event."""
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
def _listening(event: str,
               listener: Callable[..., None],
) -> Generator[None, None, None]:
    """Context manager that subscribes and unsubscribes an event listener."""
    _subscribe(event, listener)
    try:
        yield
    finally:
        _unsubscribe(event, listener)


@contextlib.contextmanager
def _extracting(event: str,
                extractor: Callable[..., _R],
) -> Generator[List[_R], None, None]:
    """Context manager that provides a list of custom-extracted event data."""
    extracts = []
    with _listening(event, lambda *args: extracts.append(extractor(*args))):
        yield extracts


# FIXME: Move as much of this class's docstring as is reasonable to the module
#        docstring, but wait to do so until the SyncHook subclass is written.
#
# FIXME: Note, somewhere, that a future version of the library *might* scale up
#        to an asymptotically faster data structure for rows containing large
#        numbers (thousands? more?) of listeners.
#
class Hook:
    """
    Audit hook wrapper that managers event-specific subscribers.

    Listeners subscribe to specific auditing events and may unsubscribe from
    them. Only one audit hook is actually installed (per Hook instance). The
    actual audit hook for a Hook instance is installed the first time a
    listener subscribes to the Hook instance, so if the Hook is never needed,
    no audit hook is installed. The suggested approach is to use only a small
    number of Hook instances, often just one, even even if many different
    listeners will be subscribed and unsubscribed for many (or a few) different
    events.

    The subscribe and unsubscribe methods are NOT thread-safe. However, so long
    as the caller ensures no data race happens between calls to either or both
    of those methods, the state of the Hook should not be corrupted... IF the
    Python implementation is CPython or otherwise supports writing a single
    reference as an atomic operation. This is to say: at least on CPython,
    segfaults or very strange behavior shouldn't happen due to an event firing,
    even if a listener is subscribing or unsubscribing at the same time.

    Hook objects is not optimized for the case of a very large number of
    listeners being subscribed to the same event at the same time. This is
    because they store each event's listeners in an immutable sequence, which
    is rebuilt each time a listener is subscribed or unsubscribed. (This is
    part of how consistent state is maintained at all time, so the installed
    audit hook doesn't need to synchronize with subscribe and unsubscribe.)
    Subscribing N listeners to the same event on the same Hook, without
    unsubscribing any, takes O(N**2) time. If you need to have more than a
    couple hundred listeners on the same event at the same time, this may not
    be the right tool for the job.
    """
    # FIXME: Implement this.
