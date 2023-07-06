import os
import argparse
import zipfile
import numpy

import zip_files


def test_algorithms(filename, keep_outputs=False):
    """Test the zip compression algorithms on a single file."""
    algorithm_names = (['deflate'] * 10) + (['bz2'] * 9) + ['lzma']
    algo_levels = list(range(10)) + list(range(1, 10)) + [0]
    assert len(algorithm_names) == len(algo_levels)

    base, ext = os.path.splitext(filename)
    znames = [base + "_{0}_{1}.zip".format(alg, lev) for (alg, lev) in zip(algorithm_names, algo_levels)]

    result_sizes = [None] * len(znames)

    for i, (aname, alevel, zname) in enumerate(zip(algorithm_names, algo_levels, znames)):
        print("{0:>2d}/{1} ".format(i + 1, len(znames)), end="")
        zip_files.zip_file_or_dir(filename,
                                  outfile=zname,
                                  algorithm=aname,
                                  compress_level=alevel,
                                  overwrite=True,
                                  delete_originals=False,
                                  verbose=True)
        # assert os.path.exists(zname)
        # assert zipfile.is_zipfile(zname)

        result_sizes[i] = os.path.getsize(zname)

    # Print results
    orig_size = os.path.getsize(filename)
    # Sort the compression levels from greatest to least.
    sorted_i = numpy.argsort(result_sizes)
    anames_sorted = numpy.take_along_axis(algorithm_names, sorted_i)
    alevels_sorted = numpy.take_along_axis(algo_levels, sorted_i)
    sizes_sorted = numpy.take_along_axis(result_sizes, sorted_i)
    print("Rankings:")
    for i, (an, al, asz) in enumerate(zip(anames_sorted, alevels_sorted, sizes_sorted)):
        print("{0:>2d}. {1:>7s} L{2:d} {3:>5.2f}% compression.".format(i + 1, an, al,
                                                                       ((orig_size - asz) / orig_size) * 100.))

    if not keep_outputs:
        for zname in znames:
            os.remove(zname)

    return


def define_and_parse_args():
    parser = argparse.ArgumentParser(
        description="Test all the compression algorithms for a file, see what levels we get." +
                    " Tests 'deflate' (0-9), 'bzip2' (1-9), and 'lzma'. By default, this overwrites any zipfiles created.")
    parser.add_argument("FILE_NAME", help="The file upon which to test the compression algorithms.")
    parser.add_argument("--keep_outputs", "-k", default=False, action="store_true",
                        help="Keep all the zipfiles generated. Default: Delete them when done.")
    return parser.parse_args()


if "__main__" == __name__:
    args = define_and_parse_args()
    test_algorithms(args.FILE_NAME, keep_outputs=args.keep_outputs)
