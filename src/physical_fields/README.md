## Front data

Fronts from:
Kim, Y. S. and Orsi, A. H.: On the Variability of Antarctic Circumpolar Current Fronts Inferred from 1992–2011 Altimetry*, Journal of
Physical Oceanography, 44, 3054–3071, https://doi.org/10.1175/JPO-D-13-0217.1, 2014.

## Surface stress

Surface_stress.nc
ossx,ossy: ocean surface stress considering the sea ice modulation
taux,tauy: ERA5 wind stress;

computation of surface stress follows Dotto et al. 2018 GRL. https://doi.org/10.1029/2018GL078607

tau = rhoa*Cd*wspd*(uwind,vwind);
tauice = rhow*Ciw*icespd*(iceuvel,icevvel);
oss = tauice*iceconc+(1-iceconc)*tau;

rhoa = 1.25 kg/m^3, air density
rhow = 1028 kg/m^3, water density
Cd = 1.25e-3, drag coefficient between air and water
Ciw = 5.5e-3, drag coefficient between sea ice and water
wspd = sqrt(uwind^2+vwind^2), surface wind speed
icespd = sqrt(iceuvel^2+icevvel^2), ice drift speed

sea ice concentration: CDRv4, https://nsidc.org/data/G02202
sea ice drift: Polar Pathfinder, https://nsidc.org/data/nsidc-0116
Pathfinder sea ice drift is not the most ideal data product but it is so far the best publicly available ice drift.
It is broadly consistent with the surface wind climatologically, which is a sign of credibility as far as I understand from Paul Holland.

Ekmanvel.nc
we_tau: Ekman upwelling and downwelling velocity driven by wind stress
we_oss: Ekman upwelling and downwelling velocity driven by ocean surface stress

we = curl(tau)/rhow/f;

## Geostrophic velocity and EKE

geovel_eke.nc
u_g, v_g: geostrophic velocity
eke: monthly EKE;
EKE = ((u_g-mean(u_g)).^2+(v_g-mean(v_g)).^2)/2;

SSH data is sea-ice corrected version provided by CPOM. It is composed by two satellite missions, Envisat (2004/05-2012/03) and Cryosat-2 (2010/07-2020/04). The data is available in montly along-track format. A gaussian 300km filter, ±3 std outliner removal and 0.5x0.25 deg interpolation is applied to grid the data. Intersatellite offset is removed using the overlapped period between two missions using the mean difference map. SSH is referenced to EIGEN6C4 geoid to obtain the dynamic ocean topography feild for the computation of geostrophic velocity (ssh2vel.m). 

EKE is calculated using mean-removed Ugeo and Vgeo.
