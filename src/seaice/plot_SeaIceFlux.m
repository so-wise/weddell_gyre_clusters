%% Initial setup

% clean up workspace
clear 
close all

% SAF
load('pf_kim.txt')
load('saccf_kim.txt')
load('saf_kim.txt')
load('sbdy_kim.txt')

% select season 
n_season = 4;

%% SIempmr

load('SIempmr_seasonal.mat')

% Plots

% Seaice package: Ocean surface freshwater flux, > 0 increases salt [kg/m^2/s]:

% 1 : OND (Spring)
% 2 : JFM (Summer)
% 3 : AMJ (Fall)
% 4 : JAS (Winter)

figPos = [199 181 1422 770];

% Let's only keep the positive flux (increasing salinity)

A = f_seasonal;
A(A>0) = 0.0;
A = abs(A);
A(A<0.2e-4)=0.0;

% make figure
figure('color', 'w', 'position', figPos)
pcolor(XC, YC, squeeze(A(:,:,n_season)))
shading flat
colormap(viridis)
caxis([0 1e-4])
colorbar
hold on
plot(pf_kim(:,1), pf_kim(:,2))
plot(saf_kim(:,1), saf_kim(:,2))
plot(sbdy_kim(:,1), sbdy_kim(:,2))

%% SIempmr

load('SFLUX_seasonal.mat')

% Plots

% Net surface salt flux into the ocean (+=down), >0 increases salinity [g/m^2/s]

% 1 : OND (Spring)
% 2 : JFM (Summer)
% 3 : AMJ (Fall)
% 4 : JAS (Winter)

figPos = [199 181 1422 770];

% Let's only keep the positive flux (increasing salinity)

A = f_seasonal;
A(A>0) = 0.0;
A = abs(A);
A(A<0.001)=0.0;

% make figure
figure('color', 'w', 'position', figPos)
pcolor(XC, YC, squeeze(A(:,:,n_season)))
shading flat
colormap(viridis)
caxis([0 1e-4])
colorbar

%% Write to NetCDF file

lon = squeeze(XC(:,1));
lat = squeeze(YC(1,:));
SIfreeze = squeeze(A(:,:,4));

% create variables
nccreate('SIfreeze_SOSE.nc','lon','Dimensions',{'lon',2160});
nccreate('SIfreeze_SOSE.nc','lat','Dimensions',{'lat',198});
nccreate('SIfreeze_SOSE.nc','SIfreeze','Dimensions',{'lon',2160,'lat',198});

% write to those variables
ncwrite('SIfreeze_SOSE.nc','lon',lon);
ncwrite('SIfreeze_SOSE.nc','lat',lat);
ncwrite('SIfreeze_SOSE.nc','SIfreeze',SIfreeze);

% attributes assocaited with variable
ncwriteatt("SIfreeze_SOSE.nc","SIfreeze","Units","[g/m^2/s]")
ncwriteatt("SIfreeze_SOSE.nc","SIfreeze","Long name","Net surface salt flux into the ocean")
ncwriteatt("SIfreeze_SOSE.nc","SIfreeze","Variable name","SIfreeze")
ncwriteatt("SIfreeze_SOSE.nc","SIfreeze","Range","[0, 0.0075]")

% global attributes
ncwriteatt("SIfreeze_SOSE.nc","/","Source","SOSE iteration 100, 2005-2010")
ncwriteatt("SIfreeze_SOSE.nc","/","Source URL","http://sose.ucsd.edu/sose_stateestimation_data_05to10.html")
ncwriteatt("SIfreeze_SOSE.nc","/","Description","JAS (Austral winter) average 2005-2010")


