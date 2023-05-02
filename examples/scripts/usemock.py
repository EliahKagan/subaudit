#!/usr/bin/env python

# SPDX-License-Identifier: 0BSD

"""Using a ``Mock`` as a listener."""

from pathlib import Path
from unittest.mock import Mock

import subaudit


def main() -> None:
    """Run the ``Mock``-using experiment."""
    listener = Mock()

    with subaudit.listening('pathlib.Path.glob', listener):
        for child in Path(__file__).absolute().parent.glob('*.py'):
            print(child)

    print()
    print(listener.mock_calls)


if __name__ == '__main__':
    main()
