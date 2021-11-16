import xarray as xr
import numpy as np

np.random.seed(0)
temperature = 15 + 8 * np.random.randn(2, 2)
lon = [[-99.83, -99.32], [-99.79, -99.23]]
lat = [[42.25, 42.21], [42.63, 42.59]]
index = [0, 1, 2, 3]

# create dataset
da = xr.DataArray(
    data=temperature,
    dims=["x", "y"],
    coords=dict(
        lon=(["x", "y"], lon),
        lat=(["x", "y"], lat),
    ),
    attrs=dict(
        description="Ambient temperature.",
        units="degC",
    ),
)
da

# make 1D array of labels
d1 = xr.DataArray(
    data=np.random.randint(low=0, high=2, size=(4,)),
    dims=["z"],
    attrs=dict(
        description="Label",
    ),
)
d1

# stack DataSet
da_stacked = da.stack(z=("x","y"))
da_stacked

# assign d1 to da_stacked
da_stacked['label'] = d1
da_stacked

# final result
da_final = da_stacked.unstack()
da_final 
