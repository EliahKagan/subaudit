#!/usr/bin/env python

"""True redirection."""

import os

_STDIN_FILENO = 0
"""File descriptor of real standard input."""


def main() -> None:
    """Run the redirection experiment."""
    old_stdin_fd = os.dup(_STDIN_FILENO)
    os.close(_STDIN_FILENO)
    try:
        with open('LICENSE', encoding='utf-8') as file:
            assert file.fileno() == _STDIN_FILENO
            result = input('Should read first line of file: ')
            print(f'Got: {result!r}')
    finally:
        os.dup2(old_stdin_fd, _STDIN_FILENO)
        os.close(old_stdin_fd)

    result = input('Should read from original stdin: ')
    print(f'Got: {result!r}')


if __name__ == '__main__':
    main()
