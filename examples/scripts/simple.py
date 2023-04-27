#!/usr/bin/env python

"""
The builtins.input/result event.

This script can be run with redirection set up by a shell to show how the
builtins.input/result audit event is only raised for interactive input.
"""

import subaudit


def main() -> None:
    """Run the simple experiment."""
    with subaudit.extracting('builtins.input/result', lambda x: x) as extracts:
        result = input('Input: ')

    # f-strings in 3.7 don't support {var=}.
    print(f'result={result!r}')
    print(f'extracts={extracts!r}')


if __name__ == '__main__':
    main()
