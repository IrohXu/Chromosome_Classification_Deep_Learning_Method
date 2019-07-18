function contourevidence = mia_cmpcontourevidence(I,k,thd1,thd2,thdn,vis)
% mia_contourevidence performs contour evidenc extraction step of the method.

%   Synopsis
%       contourevidence = mia_cmpcontourevidence(I,k,thd1,thd2,thdn,vis)
%   Description
%        Returns the contour evidences (visible parts of the objects ) 
%        to infernce the visible parts. It involves two separate tasks:
%        contour segmentation and segment grouping
%   Inputs 
%          - I           binary image
%          - k           kth adjucnet points to the corner point
%          - thd1        Euclidean distance between ellipse centroid of the 
%                        combined contour segments and ellipse fitted to each segment
%          - thd2        Euclidean distance between between the centroids of ellipse
%                        fitted to each segment.
%          - thdn        Euclidean distance between contour center points
%                        to define neighbouring segments 
%          - vis         0 or 1 for visualization puropose
%   Outputs
%         - contourevidence    a cell array contating the visile objects boundaries 

%   Authors
%          Sahar Zafari <sahar.zafari(at)lut(dot)fi>
%
%   Changes
%       14/01/2016  First Edition

    % concave point extraction
    fprintf('Extracting the concave points.....\n')
     % parameters for css method
    [C,T_angle,sig,H,L,Endpoint,Gap_size] = parse_inputs(); 
    [curve,idxconcavepoints]= mia_cmpconcavepoint_css(I,C,T_angle,sig,H,L,Endpoint,Gap_size,k,vis);
    % segment the curve by the detetcted concave points
    fprintf('Segmenting the curve by detected concave points.....\n')
    [segments,centers] = mia_segmentcurve_concave(I,curve,idxconcavepoints,vis);
    % segmnet grouping
    contourevidence = mia_groupsegments(I,segments,centers,thdn,thd1,thd2,vis);
end
