"""subaudit: Subscribe and unsubscribe on specific audit events."""

__all__ = [
    'audit',
    'addaudithook',
    'Hook',  # FIXME: Write methods: subscribe, unsubscribe, listen, extract.
]

try:
    from sys import audit, addaudithook
except ImportError:
    from sysaudit import audit, addaudithook
