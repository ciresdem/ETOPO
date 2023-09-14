import numpy
import pandas
import os

####################################3
# Include the base /src/ directory of thie project, to add all the other modules.
import import_parent_dir; import_parent_dir.import_src_dir_via_pythonpath()
####################################3
import utils.configfile
my_config = utils.configfile.config()
import icesat2.plot_validation_results
import utils.traverse_directory
import utils.sizeof_format

validation_dir = os.path.join(my_config.etopo_validation_results_directory.format(15), "2022.09.29")

output_h5 = os.path.join(validation_dir, "plots", "total_results.h5")

# # Create the total_results.h5 file.
# # Un-comment these lines out to re-create it.
# results_file_list = utils.traverse_directory.list_files(validation_dir,
#                                                         regex_match=r"\d{2}_results.h5\Z",
#                                                         depth=0)
#
# print(len(results_file_list), ".h5 files totaling",
#       utils.sizeof_format.sizeof_fmt(sum([os.path.getsize(fn) for fn in results_file_list])))
# total_df = icesat2.plot_validation_results.get_data_from_h5_or_list(results_file_list, include_filenames=True)
#
# if os.path.exists(output_h5):
#     print("Removing old", os.path.basename(output_h5))
#     os.remove(output_h5)
#
# total_df.to_hdf(output_h5, "icesat2", complevel=2, complib="zlib")
# print(os.path.basename(output_h5), "written with", len(total_df), "records.")

# If we're not creating the total_results.h5 file above, just read it.
print("Reading", os.path.basename(output_h5), "...", end=" ", flush=True)
total_df = pandas.read_hdf(output_h5)
print("Done.")

# Create the subsets of the total_results.h5 with coverage_area thresholds.
for pct in range(20, 40): #(40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50):
    df_subset = total_df[total_df["coverage_frac"] >= (pct / 100.)]
    sub_out_h5 = os.path.splitext(output_h5)[0] + f"_gte{pct}.h5"
    df_subset.to_hdf(sub_out_h5, "icesat2", complevel=2, complib="zlib")
    print(os.path.basename(sub_out_h5), "written with", len(df_subset), "records.")
