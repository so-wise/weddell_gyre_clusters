%% Initial setup

% clean up workspace
clear 
close all

% write to file flag
writeToFileFlag = true;

% SAF
load('pf_kim.txt')
load('saccf_kim.txt')
load('saf_kim.txt')
load('sbdy_kim.txt')

% select season 
n_season = 4;

%% SIempmr

load('SIempmr_seasonal.mat')

ymask = repmat(YC,[1 1 4]);

% Plots

% Seaice package: Ocean surface freshwater flux, > 0 increases salt [kg/m^2/s]:

% 1 : OND (Spring)
% 2 : JFM (Summer)
% 3 : AMJ (Fall)
% 4 : JAS (Winter)

figPos = [199 181 1422 770];

% Let's only keep the positive flux (increasing salinity)

%threshold = 0.1e-4;
threshold = 1e-5;

A = f_seasonal;
A(A>0) = 0.0;
A = abs(A);
A(A<threshold)=0.0;

% lat mask
A(ymask>=-54) = 0.0;
%A(ymask<=-65) = 0.0;

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

% %% SFLUX
% 
% load('SFLUX_seasonal.mat')
% 
% % Plots
% 
% % Net surface salt flux into the ocean (+=down), >0 increases salinity [g/m^2/s]
% 
% % 1 : OND (Spring)
% % 2 : JFM (Summer)
% % 3 : AMJ (Fall)
% % 4 : JAS (Winter)
% 
% figPos = [199 181 1422 770];
% 
% % Let's only keep the positive flux (increasing salinity)
% 
% threshold = 0.0;    % have used 0.001
% 
% A = f_seasonal;
% A(A>0) = 0.0;
% A = abs(A);
% A(A<threshold)=0.0;
% 
% % make figure
% figure('color', 'w', 'position', figPos)
% pcolor(XC, YC, squeeze(A(:,:,n_season)))
% shading flat
% colormap(viridis)
% caxis([0 1e-4])
% colorbar

%% Write to NetCDF file

if writeToFileFlag==true

    lon = squeeze(XC(:,1));
    lat = squeeze(YC(1,:));
    SIfreeze = squeeze(A(:,:,4));

    % create variables
    nccreate('SIfreeze_SOSE_R1.nc','lon','Dimensions',{'lon',2160});
    nccreate('SIfreeze_SOSE_R1.nc','lat','Dimensions',{'lat',198});
    nccreate('SIfreeze_SOSE_R1.nc','SIfreeze','Dimensions',{'lon',2160,'lat',198});

    % write to those variables
    ncwrite('SIfreeze_SOSE_R1.nc','lon',lon);
    ncwrite('SIfreeze_SOSE_R1.nc','lat',lat);
    ncwrite('SIfreeze_SOSE_R1.nc','SIfreeze',SIfreeze);

    % attributes assocaited with variable
    ncwriteatt("SIfreeze_SOSE_R1.nc","SIfreeze","Units","[kg/m^2/s]")
    ncwriteatt("SIfreeze_SOSE_R1.nc","SIfreeze","Long name","Seaice package: Ocean surface freshwater flux")
    ncwriteatt("SIfreeze_SOSE_R1.nc","SIfreeze","Variable name","SIfreeze")
    ncwriteatt("SIfreeze_SOSE_R1.nc","SIfreeze","Threshold",num2str(threshold))
    ncwriteatt("SIfreeze_SOSE_R1.nc","SIfreeze","Convention","> 0 increases salt")

    % global attributes
    ncwriteatt("SIfreeze_SOSE_R1.nc","/","Source","SOSE iteration 100, 2005-2010")
    ncwriteatt("SIfreeze_SOSE_R1.nc","/","Source URL","http://sose.ucsd.edu/sose_stateestimation_data_05to10.html")
    ncwriteatt("SIfreeze_SOSE_R1.nc","/","Description","JAS (Austral winter) average 2005-2010")

end


