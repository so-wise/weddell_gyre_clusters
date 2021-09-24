#####################################################################
# Analysis tools
#####################################################################

# import pakcages
import numpy as np
import xarray as xr

#####################################################################
# Split by longitude sectors
#####################################################################
def split_single_class_by_longitude(df, class_split, lon_split):

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
