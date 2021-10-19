import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from shapely.ops import cascaded_union

fig = plt.figure(figsize=(10, 4), dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(), aspect='equal')
ax.set_extent([-100, 10, 0, 90], crs=ccrs.PlateCarree())

# Put a background image on for nice sea rendering.
ax.stock_img()

# ticks
ax.set_xticks(np.arange(-90, 30, 30), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(40, 100, 20), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

# add land and coastline and 1000m bathymetry line
ax.add_feature(cfeature.LAND, facecolor='grey', zorder=1)
ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)
#ax.add_feature(cfeature.OCEAN, linewidth=1.0, zorder=1)

bathym = cfeature.NaturalEarthFeature(name='bathymetry_K_200', scale='10m', category='physical')
bathym = cascaded_union(list(bathym.geometries()))
ax.add_geometries(bathym, facecolor='none', edgecolor='grey', linestyle='solid', linewidth=0.5, crs=ccrs.PlateCarree())
