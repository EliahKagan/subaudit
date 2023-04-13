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
