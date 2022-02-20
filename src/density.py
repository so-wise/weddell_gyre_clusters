#####################################################################
# Utilities for calculating density w/ Gibbs Sea Water (gsw, TEOS-10)
#####################################################################

import gsw
import xarray as xr

#####################################################################
# Calculate density of each profile in an xarray dataset
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

    # apply functions
    sa = xr.apply_ufunc(gsw.SA_from_SP, sp, p, lon, lat, dask='parallelized', output_dtypes=[sp.dtype])
    ct = xr.apply_ufunc(gsw.CT_from_pt, sa, pt, dask='parallelized', output_dtypes=[sa.dtype])
    sig0 = xr.apply_ufunc(gsw.density.sigma0, sa, ct, dask='parallelized', output_dtypes=[sa.dtype])

    # add sig0 to existinng profiles_antarctic dataset
    profiles['sig0'] = sig0
    profiles['prof_SA'] = sa
    profiles['prof_CT'] = ct

    # return it
    return profiles

#####################################################################
# Calculate dynamic height anomaly, N2,
#####################################################################
def calc_geostropic_currents(profiles):

    # display
    print('density.calc_geostrophic_currents')

    # extract a few variables
    sa = profiles.prof_SA.T
    ct = profiles.prof_CT.T
    p = profiles.depth
    lon = profiles.lon
    lat = profiles.lat

    # apply function : get dynamic height
    dynamic_height = xr.apply_ufunc(gsw.geostrophy.geo_strf_dyn_height,
        sa, ct, p, dask='parallelized', output_dtypes=[sa.dtype])

    # geostrophic velocity
    #geo_vel, mid_lon, mid_lat = xr.apply_ufunc(gsw.geostrophy.geostrophic_velocity,
    #    dynamic_height, lon, lat, dask='parallelized', output_dtypes=[sa.dtype])

    #Nsquared = xr.apply_ufunc(gsw.stability.Nsquared, sa, ct, p,
    #    dask='parallelized', output_dtypes=[sa.dtype])

    return dynamic_height 

#####################################################################
# Scalar density value
#####################################################################
def calc_scalar_density(pt, sp, p, lon, lat):

    sa = gsw.SA_from_SP(sp, p, lon, lat)
    ct = gsw.CT_from_pt(sa, pt)
    sig0 = gsw.density.sigma0(sa,ct)

    return sig0
