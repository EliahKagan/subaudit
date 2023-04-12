"""subaudit: Subscribe and unsubscribe for specific audit events."""

__all__ = [
    'audit',
    'addaudithook',
    'Hook',  # FIXME: Methods: subscribe, unsubscribe, listening, extracting.
    'skip_if_unavailable',
]

try:
    from sys import audit, addaudithook
except ImportError:
    from sysaudit import audit, addaudithook
