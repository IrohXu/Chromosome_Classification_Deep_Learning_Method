function stats = mia_particles_segmentation(I,param)
% mia_particles_segmentation performs segmentation by using the concave points.
%   Synopsis
%       stats = mia_segmentation_concave(I,k,thd1,thd2,thdn)
%   Description
%         Returns segmentation result (objects boundaries) of overlapping nanoparticles
%         by using concave points and ellipse fitting propeties.
%   Inputs 
%          - I         grayscale or binary Image
%          - k         kth adjucnet points to the corner point
%         - thd1       Euclidean distance between ellipse centroid of the 
%                      combined contour segments and ellipse fitted to each segment
%         - thd2       Euclidean distance between between the centroids of ellipse
%                      fitted to each segment.
%         - thdn       Euclidean distance between contour center points
%                       to define neighbouring segments 
%        - vis1        visualize the contoure evidence extraction step
%        - vis2        visualize the contoure estimation step
%   Outputs
%         - stats      cell array contating the objects boundaries 

%         
%   Authors
%          Sahar Zafari <sahar.zafari(at)lut(dot)fi>
%
%   Changes
%       14/01/2016  First Edition

    % load the parameters

    k = param(1);
    thd1 = param(2);
    thd2 = param(3);
    thdn = param(4);
    vis1 = param(5);
    vis2 = param(6);

    % Image Binarization by otsu's method
    level = graythresh(I);
    imgbw =  ~im2bw(I,level);
    % Contour Evidence Extraction
    fprintf('Performs Contour Evidence Extraction.....\n')
    contourevidence = mia_cmpcontourevidence(imgbw,k,thd1,thd2,thdn,vis1);
    % Contour Estimation
    fprintf('Performs Contour Estimation.....\n')
    stats =  mia_estimatecontour_lsf(I,contourevidence,vis2);

end
