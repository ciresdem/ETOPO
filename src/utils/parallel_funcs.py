# -*- coding: utf-8 -*-

import sys
import os
import multiprocessing as mp
import numpy

def physical_cpu_count():
    """On this machine, get the number of physical cores.

    Not logical cores (when hyperthreading is available), but actual physical cores.
    Things such as multiprocessing.cpu_count often give us the logical cores, which
    means we'll spin off twice as many processes as really helps us when we're
    multiprocessing for performance. We want the physical cores."""
    if sys.platform == "linux" or sys.platform == "linux2":
        # On linux. The "linux2" variant is no longer used but here for backward-compatibility.
        lines = os.popen('lscpu').readlines()
        line_with_sockets = [l for l in lines if l[0:11] == "Socket(s): "][0]
        line_with_cps = [l for l in lines if l[0:20] == "Core(s) per socket: "][0]

        num_sockets = int(line_with_sockets.split()[-1])
        num_cores_per_socket = int(line_with_cps.split()[-1])

        return num_sockets * num_cores_per_socket

    elif sys.platform == "darwin":
        # On a mac
        # TODO: Flesh this out from https://stackoverflow.com/questions/12902008/python-how-to-find-out-whether-hyperthreading-is-enabled
        return mp.cpu_count()

    elif sys.platform == "win32" or sys.platform == "win64" or sys.platform == "cygwin":
        # On a windows machine.
        # TODO: Flesh this out from https://stackoverflow.com/questions/12902008/python-how-to-find-out-whether-hyperthreading-is-enabled
        return mp.cpu_count()

    else:
        # If we don't know what platform they're using, just default to the cpu_count()
        # It will only get logical cores, but it's better than nothing.
        return mp.cpu_count()

# A dictionary for converting numpy array dtypes into carray identifiers.
# For integers & floats... does not hangle character/string arrays.
# Reference: https://docs.python.org/3/library/array.html
dtypes_dict = {numpy.int8:    'b',
               numpy.uint8:   'B',
               numpy.int16:   'h',
               numpy.uint16:  'H',
               numpy.int32:   'l',
               numpy.uint32:  'L',
               numpy.int64:   'q',
               numpy.uint64:  'Q',
               numpy.float32: 'f',
               numpy.float64: 'd',
               # Repeat for these expressions of dtype as well.
               numpy.dtype('int8'):    'b',
               numpy.dtype('uint8'):   'B',
               numpy.dtype('int16'):   'h',
               numpy.dtype('uint16'):  'H',
               numpy.dtype('int32'):   'l',
               numpy.dtype('uint32'):  'L',
               numpy.dtype('int64'):   'q',
               numpy.dtype('uint64'):  'Q',
               numpy.dtype('float32'): 'f',
               numpy.dtype('float64'): 'd'}
