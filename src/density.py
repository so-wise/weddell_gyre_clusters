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
# Calculate dynamic height anomaly
#####################################################################
def calc_dynamic_height(profiles):

    # display
    print('density.calc_dynamic_height')

    # extract a few variables
    sa = profiles.prof_SA.T
    ct = profiles.prof_CT.T
    p = profiles.depth
    lon = profiles.lon
    lat = profiles.lat

    # apply function : get dynamic height
    dynamic_height = xr.apply_ufunc(gsw.geostrophy.geo_strf_dyn_height,
        sa, ct, p, dask='parallelized', output_dtypes=[sa.dtype])

    # add to Dataset
    profiles['dyn_height'] = dynamic_height

    return profiles

#####################################################################
# Buoyancy frequency (stability) [using numpy approach for now]
#####################################################################
def calc_Nsquared(df):

    # extract a few variables
    sa = df.prof_SA.values
    ct = df.prof_CT.values
    p = df.depth.values
    lon = df.lon.values
    lat = df.lat.values

    # initialize array of zeros
    Nsquared = np.zeros((sa.shape[0],sa.shape[1]-1))
    Nsquared[:,:] = np.NaN
    p_mid = np.zeros((sa.shape[1]-1))
    p_mid[:] = np.NaN

    # loop over profiles, calculate N2 separately
    for n in np.arange(sa.shape[0]):
        Nsq1, p_mid1 = gsw.stability.Nsquared(sa[n,:], ct[n,:], p)
        Nsquared[n,:] = Nsq1
        p_mid[:] = p_mid1

    # convert to DataArray
    da = xr.DataArray(data=Nsquared,
                      dims=["profile", "depth_mid"],
                      coords=dict(
                          profile=df.profile.values,
                          depth_mid=p_mid),
                      attrs=dict(
                            description="Buoyancy frequency",
                            units="1/s^2",))

    # add to df dataset
    df['Nsquared'] = da

    return df

#####################################################################
# Mixed layer depth (integral depth-scale method, numpy approach for now)
# --- see Thomson and Fine (2003, JAOT)
#####################################################################
def calc_mixed_layer_depth(df):

    # extract a few variables
    N2 = df.Nsquared.values
    p = df.depth_mid.values

    # initialize array of zeros
    mld = np.zeros((N2.shape[0],))
    mld[:] = np.NaN

    # loop over profiles, integrate downward (z_b= 1000 m reference)
    for n in np.arange(N2.shape[0]):

        N2a = N2[n,:]
        A = np.cumsum(p*N2a)/np.cumsum(N2a)
        mld[n] = A[-1]

    # convert to DataArray
    da = xr.DataArray(data=mld,
                      dims=["profile"],
                      coords=dict(profile=df.profile.values),
                      attrs=dict(description="Mixed layer depth",units="m"))

    # add to df dataset
    df['mld'] = da

    return df

#####################################################################
# Scalar density value
#####################################################################
def calc_scalar_density(pt, sp, p, lon, lat):

    sa = gsw.SA_from_SP(sp, p, lon, lat)
    ct = gsw.CT_from_pt(sa, pt)
    sig0 = gsw.density.sigma0(sa,ct)

    return sig0
