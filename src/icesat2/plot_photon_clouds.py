#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 20:04:24 2021

plot_photon_clouds.py -- Simple tool for making photon point clouds with matplotlib

@author: mmacferrin
"""

# import retrieve_land_photons
import atl_granules
import progress_bar

import pandas
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.gridspec
import matplotlib.ticker
import os
# import numpy

def plot_photon_data(photon_h5, granule_id1_int=None, beam_id=None, bbox=None, surface_field="h_geoid",
                     fix_reef=False, overwrite=False, verbose=True):

    if verbose:
        print("Reading", photon_h5, end="...")
    df = pandas.read_hdf(photon_h5)
    if verbose:
        print("done.")

    if type(beam_id) == int:
        beam_number = beam_id
        beam_name = atl_granules.beam_int_to_name(beam_number)
    elif type(beam_id) == str:
        beam_number = beam_id
        beam_name = atl_granules.beam_int_to_name(beam_number)
    elif beam_id is None:
        # By default, just literally get the first beam.
        beam_number = df.iloc[0]["beam"]
        beam_name = atl_granules.beam_int_to_name(beam_number)
    else:
        raise ValueError("Unknown Beam ID {}".format(repr(beam_id)))

    # print(df)
    # print(df.columns)

    # When we print the dataframe, include all the columns.
    pandas.set_option('display.max_columns', None)

    if granule_id1_int:
        # Subset dataframe by granule_id1 and beam
        df = df[(df["granule_id1"] == granule_id1_int)]

    df = df[df["beam"] == beam_number]

    # parray = retrieve_land_photons.get_photon_data(granule_id, beam=beam)
    # foobar
    # print(df)
    # print(df.columns)
    # print(df['beam'].unique())

    # Type codes:
    # -1 : uncoded
    #  0 : noise
    #  1 : ground
    #  2 : canopy
    #  3 : top of canopy
    #  4 : ocean surface (added after-the-fact, kinda cheating for now)
    colors={-1:"lightgrey", 0:"lightgrey", 1:"brown", 2:"mediumseagreen", 3:"green", 4:"dodgerblue"}

    labels = {-1: None, 0:"Noise", 1:"Land / Seafloor", 2:"Canopy", 3:"Canopy Top", 4: "Sea Surface"}

    # parray_sub =
    # N = 1
    # idxs = numpy.arange(0,len(dist_x_km),N)

    # print("dist_x:", df['dist_x'])
    # print("dist_x.min:", df['dist_x'].min())

    # dist_x_km = df["dist_x"] / 1e3

    # print(dist_x_km.min())

    if bbox != None:
        lats = df["latitude"]
        lons = df["longitude"]
        xmin, ymin, xmax, ymax = bbox

        subset_mask = (lons >= xmin) & (lons <= xmax) & (lats >= ymin) & (lats <= ymax)
        df = df[subset_mask]
        # dist_x_km = dist_x_km[subset_mask]
    # start=1500
    # end=1680
    # subset = (dist_x_km >= start) & (dist_x_km <end)

    # print(df)
    # print(df.columns)

    dist_x_km = (df["dist_x"] - df["dist_x"].min())/1e3

    # Define the plot
    DPI = 500

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.0,3.5), dpi=DPI) #, wspace=1, hspace=1)
    # ASPECT = 2.0
    # ax1.set_aspect(ASPECT)
    # ax2.set_aspect(ASPECT*0.25)
    # fig = plt.Figure(figsize=(10.0,6.5), dpi=DPI)
    # gs1 = matplotlib.gridspec.GridSpec(2, 1)
    # ax1 = fig.add_subplot(gs1[0])
    # ax2 = fig.add_subplot(gs1[1])

    # ax1 = fig.add_subplot(2,1,1)
    # ax2 = fig.add_subplot(2,1,2)

    class_id_list = [-1,0,1,2,3] #,4] # sorted(list(colors.keys()))[1:]
    for class_id in class_id_list:
        color = colors[class_id]
        label = labels[class_id]

        if class_id == 1:
            zorder=2
        else:
            zorder=1

        if class_id in [-1,0]:
            markersize=1
            marker = '.'
        else:
            markersize=1
            marker = "o"


        color_mask = (df["class_code"] == class_id)

        # plt.scatter(dist_x_km[subset] - start, df1l['height'][subset], c=df1l["class_flag"][subset].map(colors), marker=".", s=1)
        # plt.scatter(dist_x_km, parray['height'], c=numpy.vectorize(colors.__getitem__)(parray["class_code"]), marker="pixel") #, s=0.25)
        ax.scatter(dist_x_km[color_mask], df[surface_field][color_mask], c=color, label=label, marker=marker, zorder=zorder, s=markersize, linewidths=0)

    lgnd = ax.legend(loc="upper right", fontsize="small", labelspacing=0.5)

    #change the marker size manually for both lines
    legend_markersize = 20
    for color_id in range(len(class_id_list[1:])):
        lgnd.legendHandles[color_id]._sizes = [legend_markersize]
    # lgnd.legendHandles[1]._sizes = [legend_markersize]
    # lgnd.legendHandles[2]._sizes = [legend_markersize]
    # lgnd.legendHandles[3]._sizes = [legend_markersize]
    # lgnd.legendHandles[4]._sizes = [legend_markersize]


    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("{} Elevation (m)".format({"h_ellipsoid": "WGS84", "h_geoid": "Geoid", "h_meantide":"Mean Tide"}[surface_field]))

    # TODO: Get the highest & lowest land/canopy elevations. Use those.
    ax.set_ylim(-20, 150)
    # ax2.set_xlabel(ax1.get_xlabel())
    # ax2.set_ylabel(ax1.get_ylabel())
    date_str = str(df['granule_id1'].iloc[0])[0:8]
    rgt_str = int(str(df['granule_id2'].iloc[0]).zfill(13)[0:4])
    # ax = plt.gca()
    plt.text(0.02,0.98,"New England, United States\n{0}.{1}.{2}\nRGT #{3}\nLaser {4}".format(
        date_str[0:4], date_str[4:6], date_str[6:8],
        rgt_str,
        beam_name[-2:].upper()),
        ha="left", va="top", fontsize="large", transform=ax.transAxes)
    # ax1.text(0.02,0.02,"South", ha="left", va="bottom", fontsize="large", transform=ax1.transAxes)
    # ax1.text(0.98,0.02,"North", ha="right", va="bottom", fontsize="large", transform=ax1.transAxes)
    # ax1.set_ylim((-15,27))

    # ax2.set_xlim(subset_box[0])
    # ax2.set_ylim(subset_box[1])

    # ax2.xaxis.grid(True, which='major', lw=0.25, zorder=-1)
    # ax2.xaxis.grid(True, which='minor', lw=0.1, zorder=-1)
    # ax2.yaxis.grid(True, which='major', lw=0.25, zorder=-1)
    # ax2.yaxis.grid(True, which='minor', lw=0.1, zorder=-1)
    # ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    # ax2.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    # ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    # ax2.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    # ax2.set_xticklabels(ax2.get_xticks(), rotation=90)

    # fig.tight_layout()

    # print(parray)
    # print(parray.dtype)
    # print(parray.shape)

    plt.tight_layout()
    figname = os.path.splitext(photon_h5)[0] + "_" + str(granule_id1_int) + "_" + beam_name + ".png"
    plt.savefig(figname) #, dpi=DPI)
    print(figname, "written.")

    return df


def plot_photon_data_gbr(photon_h5, granule_id1_int=None, beam_id=None, bbox=None, surface_field="h_geoid",
                     fix_reef=False, overwrite=False, verbose=True):

    if verbose:
        print("Reading", photon_h5, end="...")
    df = pandas.read_hdf(photon_h5)
    if verbose:
        print("done.")

    if type(beam_id) == int:
        beam = atl_granules.ATL_granule.beam_code_dict[beam_id]
    elif type(beam_id) == str:
        beam = beam_id
    elif beam_id is None:
        beam = atl_granules.beam_int_to_name(df.iloc[0]["beam"])
    else:
        raise ValueError("Unknown Beam ID {}".format(repr(beam_id)))

    # print(df)
    # print(df.columns)

    # When we print the dataframe, include all the columns.
    pandas.set_option('display.max_columns', None)

    if granule_id1_int:
        # Subset dataframe by granule_id1 and beam
        df = df[(df["granule_id1"] == granule_id1_int)]

    if beam_id:
        beam_number = atl_granules.beam_name_to_int(beam_id)
        df = df[df["beam"] == beam_number]

    # parray = retrieve_land_photons.get_photon_data(granule_id, beam=beam)
    # foobar
    # print(df)
    # print(df.columns)
    # print(df['beam'].unique())

    # Type codes:
    # -1 : uncoded
    #  0 : noise
    #  1 : ground
    #  2 : canopy
    #  3 : top of canopy
    #  4 : ocean surface (added after-the-fact, kinda cheating for now)
    colors={-1:"lightgrey", 0:"lightgrey", 1:"brown", 2:"mediumseagreen", 3:"green", 4:"dodgerblue"}

    labels = {-1: None, 0:"Noise", 1:"Land / Seafloor", 2:"Canopy", 3:"Canopy Top", 4: "Sea Surface"}

    # parray_sub =
    # N = 1
    # idxs = numpy.arange(0,len(dist_x_km),N)

    # print("dist_x:", df['dist_x'])
    # print("dist_x.min:", df['dist_x'].min())

    # dist_x_km = df["dist_x"] / 1e3

    # print(dist_x_km.min())

    if bbox != None:
        lats = df["latitude"]
        lons = df["longitude"]
        xmin, ymin, xmax, ymax = bbox

        subset_mask = (lons >= xmin) & (lons <= xmax) & (lats >= ymin) & (lats <= ymax)
        df = df[subset_mask]
        # dist_x_km = dist_x_km[subset_mask]
    # start=1500
    # end=1680
    # subset = (dist_x_km >= start) & (dist_x_km <end)

    # print(df)
    # print(df.columns)

    dist_x_km = (df["dist_x"] - df["dist_x"].min())/1e3

    # print(dist_x_km.min())

    # print(df['dist_x'].min())
    # foobar

    # FOR THIS GREAT_BARRIER_REEF CLOUD, change some of the photon classifications from canopy-->noise, and ground--ocean
    # Change the ocean-surface "ground" photons to "ocean surface"
    ground_to_ocean_surface_km_cutoff = 46.5
    slope_target = 0.36
    buffer_size = 0.50
    sea_surface = (dist_x_km - ground_to_ocean_surface_km_cutoff) * slope_target / (dist_x_km.max() - ground_to_ocean_surface_km_cutoff)

    # If we haven't yet reclassified the photons, do so here.
    if 'class_code_rc' not in list(df.columns) or overwrite:
        # Create a column for reclassified photon codes. Keep the originals as-is.
        df['class_code_rc'] = df['class_code'].copy()

        # Canopy Photons outside of SS +- buffer --> noise
        #  Only over the ocean
        # Classified as canopy
        # Outside of buffer zone.
        # Turn into noise
        df.loc[(dist_x_km >= ground_to_ocean_surface_km_cutoff) & \
                df['class_code_rc'].between(2,3, inclusive='both') & \
               ((df[surface_field] < (sea_surface-buffer_size)) | (df[surface_field] > (sea_surface+buffer_size))),
                   "class_code_rc"] = 0


        # Canopy inside the buffer zone: turn into sea-surface.
        # Only over the ocean
        # Classified as canopy
        # Inside the buffer zone
        # Turn into sea-surface
        df.loc[(dist_x_km >= ground_to_ocean_surface_km_cutoff) & \
                df['class_code_rc'].between(2,3, inclusive='both') & \
                df[surface_field].between((sea_surface-buffer_size), (sea_surface+buffer_size), inclusive="both"),
                   "class_code_rc"] = 4

        # Now for the hard part. Go through by increment (0.1 km?).
        # For each increment:
        #     If no sea-surface within that increment, then color ground within the buffer --> sea-surface.
        STEP_SIZE_KM = 0.05
        increment_boundaries = numpy.arange(ground_to_ocean_surface_km_cutoff, dist_x_km.max(), STEP_SIZE_KM)
        ss_photons = (df['class_code_rc'] == 4)
        gr_photons = (df['class_code_rc'] == 1)

        # Create a "change_to_ss" field to help skip this (time-consuming) step in future calculations.
        # if "change_to_ss" not in list(df.columns)):
            # df["change_to_ss"] = False

        for i,b_start in enumerate(increment_boundaries):
            # Show the progress
            progress_bar.ProgressBar(i,len(increment_boundaries), suffix="{0}/{1}".format(i,len(increment_boundaries)))

            b_end = b_start + STEP_SIZE_KM
            photons_in_step = dist_x_km.between(b_start, b_end, inclusive="left")
            if photons_in_step.sum() == 0:
                continue

            if (photons_in_step & ss_photons).sum() <= 1:
                # If one photon or less is classified as "sea-surface", then change all ground photons (within or above the buffer zone) in that segment to sea-surface.
                subset_photons = photons_in_step & gr_photons & (df[surface_field] >= (sea_surface-(buffer_size*1.1)))
                # df.loc[subset_photons, 'change_to_ss'] = True
                df.loc[subset_photons, 'class_code_rc'] = 4

                gr_photons_step = (photons_in_step & (df['class_code_rc'] == 1))
                # Sometimes there's a straggling noise ground photon or few, just outside the buffer. Get them too.
                if gr_photons_step.sum() <= 3:
                    df.loc[gr_photons_step, 'class_code_rc'] = 4

        progress_bar.ProgressBar(len(increment_boundaries),len(increment_boundaries), suffix="{0}/{1}".format(len(increment_boundaries),len(increment_boundaries)))

        base, ext = os.path.splitext(photon_h5)
        if base[-3:] != "_rc":
            fname_out = base + "_rc" + ext
        else:
            fname_out = photon_h5
        df.to_hdf(fname_out, "icesat2")
        print(fname_out, "written.")

    # Fill missing gaps in sea surface
    df.loc[(dist_x_km.between(163, 175) | dist_x_km.between(220, 228) | dist_x_km.between(258.5, 297)) &
           df[surface_field].between(sea_surface - buffer_size, sea_surface + buffer_size),
           "class_code_rc"] = 4
    # Fill missing reef floor ground photons! Important stuff!
    def line(xvals, x1, x2, y1, y2):
        m = (y2 - y1)/(x2 - x1)
        b = y1 - (m*x1)
        return (xvals * m) + b

    if fix_reef:

        h = df[surface_field]
        # Hand-pick out some reefs in the bounding box (very track-specific here, only 2020.04.20, RGT 375, Laser 1R

        # Reef 1, p1
        x1, x2, y1, y2 = (233.65, 233.8, -0.8, -4.0)
        buffer = 0.4
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1
        # Reef 1, p2
        x1, x2, y1, y2 = (233.75, 233.8, -1, -3)
        buffer = 0.5
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1

        # Reef 2, p0
        x1, x2, y1, y2 = (237.05, 237.15, -5.4, -2.5)
        buffer = 0.5
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1
        # Reef 2, p1
        x1, x2, y1, y2 = (237.1, 237.8, -4, -1.6)
        buffer = 0.5
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1
        # Reef 2, p2
        x1, x2, y1, y2 = (237.8, 238.1, -1.6, -3.4)
        buffer = 0.65
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1
        # Reef 2, p3
        x1, x2, y1, y2 = (238.1, 238.25, -2.7, -2.4)
        buffer = 0.5
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1
        # Reef 2, p4
        x1, x2, y1, y2 = (238.25, 238.4, -2.2, -0.5)
        buffer = 0.5
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1
        # Reef 2, p5
        x1, x2, y1, y2 = (239.06, 239.15, -2.3, -10)
        buffer = 0.9
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1
        # Reef 2, p6
        x1, x2, y1, y2 = (239.15, 239.2, -10, -11)
        buffer = 0.7
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1
        # Reef 2, p6
        x1, x2, y1, y2 = (239.2, 239.3, -10.3, -10.1)
        buffer = 0.7
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1
        # Reef 2, p7
        x1, x2, y1, y2 = (239.3, 239.5, -11.5, -8.5)
        buffer = 0.7
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1
        # Reef 2, p8
        x1, x2, y1, y2 = (239.45, 239.7, -9, -13)
        buffer = 0.7
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1
        # Reef 2, p9
        x1, x2, y1, y2 = (239.7, 239.8, -13, -12.8)
        buffer = 0.6
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1
        # Reef 2, p10
        x1, x2, y1, y2 = (239.8, 239.9, -12.5, -8.5)
        buffer = 0.7
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1
        # Reef 2, p11
        x1, x2, y1, y2 = (239.9, 240.0, -10.5, -9.3)
        buffer = 0.7
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1
        # Reef 2, p12
        x1, x2, y1, y2 = (240.0, 240.07, -9.5, -10.6)
        buffer = 0.7
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1
        # Reef 2, p13
        x1, x2, y1, y2 = (240.07, 240.18, -10.4, -1.4)
        buffer = 0.8
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1

        # Reef 2, water surface
        x1, x2 = (238.2, 239.15)
        df.loc[(dist_x_km.between(x1, x2) & h.between(sea_surface - 0.5, sea_surface + 0.5)), 'class_code_rc'] = 4

        # Reef 3, p1
        x1, x2, y1, y2 = (241.4, 241.45, -5.5, -1.4)
        buffer = 0.8
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1
        # Reef 3, p2
        x1, x2, y1, y2 = (241.4, 241.45, -5.5, -1.4)
        buffer = 0.8
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1
        # Reef 3, p3
        x1, x2, y1, y2 = (241.4, 241.45, -5.5, -1.4)
        buffer = 0.8
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1
        # Reef 3, p4
        x1, x2, y1, y2 = (241.45, 241.55, -2, -1.2)
        buffer = 0.8
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer)), 'class_code_rc'] = 1
        # Reef 3, p5
        x1, x2, y1, y2 = (241.55, 242.1, -0.5, -0.3)
        buffer = 0.4
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1

        # Reef 4, p1
        x1, x2, y1, y2 = (243.55, 244, -1, -0.5)
        buffer = 0.4
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1

        # Reef 5, bit
        x1, x2, y1, y2 = (244.88, 245.01, -3, -3)
        buffer = 0.7
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1

        # Reef 9, p1
        x1, x2, y1, y2 = (259.17, 259.21, -1.2, -2.6)
        buffer = 0.7
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1
        # Reef 9, p2
        x1, x2, y1, y2 = (259.2, 259.55, -2.5, -0.3)
        buffer = 0.7
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1
        # Reef 9, p3
        x1, x2, y1, y2 = (259.6, 260.5, -0.7, -0.5)
        buffer = 0.45
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1
        # Reef 9, p4
        x1, x2, y1, y2 = (260.5, 261.2, -0.5, -0.8)
        buffer = 0.5
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1
        # Reef 9, p5
        x1, x2, y1, y2 = (261.2, 262.0, -1, -1)
        buffer = 0.9
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1
        # Reef 9, p6
        x1, x2, y1, y2 = (262.0, 262.85, -0.85, -0.9)
        buffer = 0.6
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1
        # Reef 9, p7
        x1, x2, y1, y2 = (262.6, 262.85, -1.5, -3.5)
        buffer = 0.7
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1
        # Reef 9, p8
        x1, x2, y1, y2 = (262.75, 263.27, -1.5, -10)
        buffer = 1.0
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1
        # Reef 9, p9
        x1, x2, y1, y2 = (263.3, 263.41, -9.9, -10)
        buffer = 0.8
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1

        # Reef 10, p1
        x1, x2, y1, y2 = (264.93, 265.1, -13.5, -10)
        buffer = 1.0
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1
        # Reef 10, p2
        x1, x2, y1, y2 = (265.05, 265.2, -10.5, -3)
        buffer = 1.0
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1
        # Reef 10, p3
        x1, x2, y1, y2 = (265.37, 265.88, -0.6, -1.1)
        buffer = 0.75
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1
        # Reef 10, p4
        x1, x2, y1, y2 = (265.85, 266.1, -1.8, -0.5)
        buffer = 0.75
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1
        # Reef 10, p5
        x1, x2, y1, y2 = (266.1, 266.2, -0.7, -1.8)
        buffer = 0.75
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1
        # Reef 10, p6
        x1, x2, y1, y2 = (266.2, 266.3, -1.5, -0.5)
        buffer = 0.8
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1
        # Reef 10, p7
        x1, x2, y1, y2 = (266.3, 266.6, -0.7, -0.75)
        buffer = 0.6
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1
        # Reef 10, p8
        x1, x2, y1, y2 = (266.6, 266.8, -0.75, -3.5)
        buffer = 0.6
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1
        # Reef 10, p9
        x1, x2, y1, y2 = (266.75, 266.85, -2.2, -2.2)
        buffer = 0.75
        line1 = line(dist_x_km, x1, x2, y1, y2)
        df.loc[(dist_x_km.between(x1, x2) & h.between(line1-buffer, line1+buffer) & df["class_code_rc"].between(-1,0,inclusive="both")), 'class_code_rc'] = 1


    # df.loc[(dist_x_km >= ground_to_ocean_surface_km_cutoff) & (df[surface_field] > (sea_surface-buffer_size)) & (df[surface_field] < (sea_surface+buffer_size)) & df['class_code'].between(2,3, inclusive='both'), "class_code",] = 4
    # Change the remaining veg photons to noise
    # canopy_to_noise_km_cutoff = 170.2
    # df.loc[(dist_x_km >= canopy_to_noise_km_cutoff) & df['class_code'].between(2, 3, inclusive='both'), "class_code"] = 0
    # On the inset, pick up some of the bathy surface returns that aren't captured here.
    # df.loc[dist_x_km.between(356, 356.2) & df[surface_field].between(-4, -1), "class_code"] = 1
    # df.loc[dist_x_km.between(359.35, 360.7) & df[surface_field].between(-5,-1.1), "class_code"] = 1

    # Define the plot
    DPI = 500

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12.0,6.5), dpi=DPI) #, wspace=1, hspace=1)
    fig.subplots_adjust(top=0.99, bottom=0.1, wspace=0.1, hspace=0.2)
    ax1, ax2 = axes
    # ASPECT = 2.0
    # ax1.set_aspect(ASPECT)
    # ax2.set_aspect(ASPECT*0.25)
    # fig = plt.Figure(figsize=(10.0,6.5), dpi=DPI)
    # gs1 = matplotlib.gridspec.GridSpec(2, 1)
    # ax1 = fig.add_subplot(gs1[0])
    # ax2 = fig.add_subplot(gs1[1])

    # ax1 = fig.add_subplot(2,1,1)
    # ax2 = fig.add_subplot(2,1,2)

    class_id_list = [-1,0,1,2,3,4] # sorted(list(colors.keys()))[1:]
    for class_id in class_id_list:
        color = colors[class_id]
        label = labels[class_id]

        if class_id == 1:
            zorder=2
        else:
            zorder=1

        if class_id in [-1,0]:
            markersize=1
            marker = '.'
        else:
            markersize=1
            marker = "o"


        color_mask = (df["class_code_rc"] == class_id)

        # plt.scatter(dist_x_km[subset] - start, df1l['height'][subset], c=df1l["class_flag"][subset].map(colors), marker=".", s=1)
        # plt.scatter(dist_x_km, parray['height'], c=numpy.vectorize(colors.__getitem__)(parray["class_code"]), marker="pixel") #, s=0.25)
        ax1.scatter(dist_x_km[color_mask], df[surface_field][color_mask], c=color, label=label, marker=marker, zorder=zorder, s=markersize, linewidths=0)
        ax2.scatter(dist_x_km[color_mask], df[surface_field][color_mask], c=color, label=label, marker=marker, zorder=zorder, s=markersize, linewidths=0)

    lgnd = ax1.legend(loc="upper right", fontsize="small", labelspacing=0.5)

    #change the marker size manually for both lines
    legend_markersize = 20
    for color_id in range(len(class_id_list[1:])):
        lgnd.legendHandles[color_id]._sizes = [legend_markersize]
    # lgnd.legendHandles[1]._sizes = [legend_markersize]
    # lgnd.legendHandles[2]._sizes = [legend_markersize]
    # lgnd.legendHandles[3]._sizes = [legend_markersize]
    # lgnd.legendHandles[4]._sizes = [legend_markersize]

    subset_box = ((231.2, 268.5), (-14, 3))

    # Add the inset rectangle
    rect = matplotlib.patches.Rectangle((subset_box[0][0], subset_box[1][0]),
                                        subset_box[0][1] - subset_box[0][0],
                                        subset_box[1][1] - subset_box[1][0],
                                        linewidth=1,
                                        edgecolor="blue",
                                        facecolor='none')
    ax1.add_patch(rect)

    ax1.set_xlabel("Distance (km)")
    ax1.set_ylabel("{} Elevation (m)".format({"h_ellipsoid": "WGS84", "h_geoid": "Geoid", "h_meantide":"Mean Tide"}[surface_field]))
    ax2.set_xlabel(ax1.get_xlabel())
    ax2.set_ylabel(ax1.get_ylabel())
    date_str = str(df['granule_id1'].iloc[0])[0:8]
    rgt_str = int(str(df['granule_id2'].iloc[0]).zfill(13)[0:4])
    # ax = plt.gca()
    plt.text(0.25,0.98,"Great Barrier Reef, Australia\n{0}.{1}.{2}\nRGT #{3}\nLaser {4}".format(
        date_str[0:4], date_str[4:6], date_str[6:8],
        rgt_str,
        beam[-2:].upper()),
        ha="left", va="top", fontsize="large", transform=ax1.transAxes)
    ax1.text(0.02,0.02,"South", ha="left", va="bottom", fontsize="large", transform=ax1.transAxes)
    ax1.text(0.98,0.02,"North", ha="right", va="bottom", fontsize="large", transform=ax1.transAxes)
    ax1.set_ylim((-15,27))

    ax2.set_xlim(subset_box[0])
    ax2.set_ylim(subset_box[1])

    ax2.xaxis.grid(True, which='major', lw=0.25, zorder=-1)
    ax2.xaxis.grid(True, which='minor', lw=0.1, zorder=-1)
    ax2.yaxis.grid(True, which='major', lw=0.25, zorder=-1)
    ax2.yaxis.grid(True, which='minor', lw=0.1, zorder=-1)
    # ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    # ax2.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    # ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    # ax2.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    # ax2.set_xticklabels(ax2.get_xticks(), rotation=90)

    # fig.tight_layout()

    # print(parray)
    # print(parray.dtype)
    # print(parray.shape)

    figname = "../plots/GBR2_{}_{}_{}.png".format(df['granule_id1'].iloc[0], df['granule_id2'].iloc[0], beam)
    plt.savefig(figname) #, dpi=DPI)
    print(figname, "written.")

    return df


if __name__ == "__main__":
    # df = plot_photon_data("../data/great_barrier_reef_2/gbr2_photons.h5", 20200420033024, 'gt1r', bbox=[142, -14.9, 146, -12.0])
    # df = plot_photon_data("../data/great_barrier_reef_2/gbr2_photons_20200420033024.h5", bbox=[142, -14.9, 146, -12.0])
    # df = plot_photon_data("../data/great_barrier_reef_2/gbr2_photons_20200420033024_gt1r_rc.h5", overwrite=False, fix_reef = True) #, bbox=[142, -14.9, 146, -12.0])
    # plot_photon_data("ATL08_20200901235954_10480807_004_01.h5", 'gt2l', bbox=(-79.03, 26.3, -76.91, 27.05), surface="mean_tide")
    # plot_photon_data("ATL08_20200901235954_10480807_004_01.h5", 'gt1l', bbox=(-79.03,25.82,-76.91,26.95))

    df = plot_photon_data("../data/temp/ne_dems_copernicus/NE_photons.h5", granule_id1_int=20200601153829, beam_id=0, bbox=[-71.1, 41.4, -70.7, 42.8])
