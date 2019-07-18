% addpath
config
% load the particles image
imggry = imread('demo.png');
% load the method parameters
param = readparam();
fprintf('Segmentation of Overlapping Nanoparticles Using Concavepoints.....\n')
fprintf('Process started.....\n')
tic
% call segmentation function
stats = mia_particles_segmentation(imggry,param); 
toc

             
