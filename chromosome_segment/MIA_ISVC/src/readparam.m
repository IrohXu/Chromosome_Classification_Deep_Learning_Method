function [Paramethod] = readparam(varargin);
%          - k         kth adjucnet points to the corner point
%         - thd1       Euclidean distance between ellipse centroid of the 
%                      combined contour segments and ellipse fitted to each segment.
%         - thd2       Euclidean distance between between the centroids of ellipse
%                      fitted to each segment.
%         - thdn       Euclidean distance between contour center points
%                       to define neighbouring segments 
%         - vis1       0 or 1, to visualize the contour evidenc extraction
%                       step
%         - vis2       0 or 1, to visualize the contour estimation
%                       step
Paramethod=[15,23,45,90,1,1];
