% Visual angle calculation
% based on visualAngle_binSize.m
% Marleen Haupt
% 28/01/2022

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUB EEG details
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% code from: https://osdoc.cogsci.nl/3.2/visualangle/
h = 15; % Monitor height in cm
d = 60; % Distance between monitor and participant in cm
r = 1080; % Vertical resolution of the monitor
size_in_px = 223; %the stimulus size in pixels

% calculate the number of degrees that correspond to a single pixel. This will
% generally be a very small value, something like 0.03.
deg_per_px = rad2deg(atan2(.5*h, d)) / (.5*r);
pix_per_deg2=1/deg_per_px;

% calculate the size of the stimulus in degrees
size_in_deg = size_in_px * deg_per_px;

% double-check with result from this website: 
% https://michtesar.github.io/visual_angle_calculator/

% calculate pixels per degree of visual angle
pix_per_deg=size_in_px/size_in_deg;
pix_per_deg2=1/deg_per_px;

