#####################################################################
# Analysis tools
#####################################################################

# import pakcages
import numpy as np
import xarray as xr
import plot_tools as pt

#####################################################################
# Select profiles within a rectangular box
#####################################################################
def split_single_class_by_box(df, class_split, box_edges):
# split_single_class_by_box(df, class_split, box_edges)
#
# - Splits a class into those in a box and those not in the box
#   df : xarray Dataset to be split
#   class_split : which class will be split into sub-classes
#   box_edges : [lon_min, lon_max, lat_min, lat_max]
#

    # display which function is being used
    print('analysis.split_single_class_by_box')

    # first, just isolate the single class
    df = df.where(df.label==class_split, drop=True)

    # boolean mask with conditions based on box coordinates
    within_box = (df.lon >= box_edges[0]) & \
                 (df.lon <= box_edges[1]) & \
                 (df.lat >= box_edges[2]) & \
                 (df.lat <= box_edges[3])

    # use the "where" command to select "in box" and "not in box"
    # NOTE: if this crashes, it's probably because box is empty
    df_in = df.where(within_box, drop=True)
    df_out = df.where(~within_box, drop=True)

    # new variable indicating inside or outside box
    df_in['in_box'] = 'yes'
    df_out['in_box'] = 'no'

    # group together into single Dataset
    # ISSUE: this isn't working for some reason.
    # It runs, but the resulting dataset can't be split back out
    # even using df.sel
    #df1 = xr.concat([df_in,df_out], dim='in_box')

    # include the box edges for future analysis
    df_in['box_edges'] = box_edges
    df_out['box_edges'] = box_edges

    return df_in, df_out

#####################################################################
# Split by longitude only
#####################################################################
def split_single_class_by_longitude(df, class_split, lon_split):
# split_single_class_by_longitude(df, class_split, lon_split)
#   df : xarray Dataset to be split
#   class_split : which class will be split into sub-classes
#   lon_split : which longitude should we cut the classes by

    print('analysis.split_by_longitude')

    # lon splitter
    lon_oneclass = df.lon[df.label==class_split]

    # split CT by longitude
    CT_oneclass = df.prof_CT[df.label==class_split]
    CT1 = CT_oneclass[lon_oneclass <= lon_split]
    CT2 = CT_oneclass[lon_oneclass >  lon_split]
    CT1['subgroup'] = 'a'
    CT2['subgroup'] = 'b'
    CT = xr.concat([CT1,CT2],dim='subgroup')

    # split SA by longitude
    SA_oneclass = df.prof_SA[df.label==class_split]
    SA1 = SA_oneclass[lon_oneclass <= lon_split]
    SA2 = SA_oneclass[lon_oneclass >  lon_split]
    SA1['subgroup'] = 'a'
    SA2['subgroup'] = 'b'
    SA = xr.concat([SA1,SA2],dim='subgroup')

    # split density by longitude
    sig0_oneclass = df.sig0[df.label==class_split]
    sig01 = sig0_oneclass[lon_oneclass <= lon_split]
    sig02 = sig0_oneclass[lon_oneclass >  lon_split]
    sig01['subgroup'] = 'a'
    sig02['subgroup'] = 'b'
    sig0 = xr.concat([sig01,sig02],dim='subgroup')

    # split CT on sig0 by longitude
    CT_onsig_oneclass = df.ct_on_sig0[df.label==class_split]
    CT_onsig1 = CT_onsig_oneclass[lon_oneclass <= lon_split]
    CT_onsig2 = CT_onsig_oneclass[lon_oneclass >  lon_split]
    CT_onsig1['subgroup'] = 'a'
    CT_onsig2['subgroup'] = 'b'
    CT_onsig = xr.concat([CT_onsig1,CT_onsig2],dim='subgroup')

    # split SA on sig0 by longitude
    SA_onsig_oneclass = df.sa_on_sig0[df.label==class_split]
    SA_onsig1 = SA_onsig_oneclass[lon_oneclass <= lon_split]
    SA_onsig2 = SA_onsig_oneclass[lon_oneclass >  lon_split]
    SA_onsig1['subgroup'] = 'a'
    SA_onsig2['subgroup'] = 'b'
    SA_onsig = xr.concat([SA_onsig1,SA_onsig2],dim='subgroup')

    # combine into dataset
    df1 = xr.Dataset({'CT':CT, 'SA':SA, 'sig0':sig0, 'CT_onsig':CT_onsig, 'SA_onsig':SA_onsig})

    return df1

#####################################################################
# Plot profile statistics grouped by label/class and year
#####################################################################
def examine_prof_stats_by_label_and_year(ploc, df, frac = 0.95, \
                                         zmin=20, zmax=1000, \
                                         Tmin = -1.9, Tmax = 7.0, \
                                         Smin = 33.0, Smax = 35.0, \
                                         sig0min = 27.0, sig0max = 28.0, \
                                         alpha=0.1):

    print('analysis.examine_prof_stats_by_label_and_year')

    # Use original profile dataset for interannual variability
    yearMin = int(df.year.values.min())
    yearMax = int(df.year.values.max())
    labelMax = int(df.label.values.max())

    # Loop over classes and years
    for k in range(0, labelMax + 1):

        print('analysis.examine_prof_stats_by_label_and_year : class ' + str(k))

        for yr in range(yearMin, yearMax + 1):

            print('analysis.examine_prof_stats_by_label_and_year : year ' + str(yr))

            # construct the mask
            myMask = (df.year==yr) & (df.label==k)

            # only plot if there are profiles in this set
            if (myMask.max()==True):

                # use mask to select profiles
                df1y = df.where(myMask, drop=True)

                # dynamic directory (where the plots are located)
                modPloc = ploc + '/stats/K' + str(k) + '/'
                modStr = str(yr)

                # plot profiles and statistics for this year and class
                pt.plot_many_profiles(modPloc, df1y, frac=frac, \
                                      zmin=zmin, zmax=zmax, \
                                      Tmin=Tmin, Tmax=Tmax, \
                                      Smin=Smin, Smax=Smax, \
                                      sig0min=sig0min, sig0max=sig0max, \
                                      alpha=alpha, modStr=modStr)
