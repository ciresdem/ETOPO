# -*- coding: utf-8 -*-

"""Code for returning human-readable string of file sizes (in bytes)."""

def sizeof_fmt(num, suffix="B", decimal_digits=1):
    """Resturn a filesize in human readable format."""
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{num:0.{decimal_digits}f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"
