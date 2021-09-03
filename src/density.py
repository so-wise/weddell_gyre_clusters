import gsw
import xarray as xr

#####################################################################
# Calculate density using Gibbs Sea Water (gsw, TEOS-10)
#####################################################################
def calc_density(profiles):

    # display
    print('density.calc_density')

    # extract a few variables
    pt = profiles.prof_T
    sp = profiles.prof_S
    p = profiles.depth
    lon = profiles.lon
    lat = profiles.lat

    sa = xr.apply_ufunc(gsw.SA_from_SP, sp, p, lon, lat, dask='parallelized', output_dtypes=[sp.dtype])
    ct = xr.apply_ufunc(gsw.CT_from_pt, sa, pt, dask='parallelized', output_dtypes=[sa.dtype])
    sig0 = xr.apply_ufunc(gsw.density.sigma0, sa, ct, dask='parallelized', output_dtypes=[sa.dtype])

    # add sig0 to existinng profiles_antarctic dataset
    profiles['sig0'] = sig0
    profiles['prof_SA'] = sa
    profiles['prof_CT'] = ct

    # return it
    return profiles
