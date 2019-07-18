function dist = mia_cmpdistance(par1, par2, par12,thd1,thd2)
% mia_cmpdistance  computes the distance betwwen fitted ellipses.
%   Synopsis
%       dist = mia_cmpdistance(par1, par2, par12,thd1,thd2)
%   Description
%         Returns the distance as 0 or 1, 0 if the two contour segments
%         whose ellipse models are at very far distance from each other.
%         or  ellipse fitted to the combined contour segments at far distance
%         from the ellipses fitted to each individual
%         contour segments.
%   Inputs 
%          - par1       elipse paramters of first segmet
%          - par2       elipse paramters of second segmet
%          - par12      elipse paramters of combined segmet
%         -  thd1       Euclidean distance between ellipse centroid of the 
%                      combined contour segments and ellipse fitted to each segment.
%         -  thd2       Euclidean distance between between the centroids of ellipse
%                      fitted to each segment
%   Outputs
%         - dist  0 or 1, 1 if the two segments should be combined.
%         
%   Authors
%          Sahar Zafari <sahar.zafari(at)lut(dot)fi>
%
%   Changes
%       14/01/2016  First Edition


    x1 = par1(3); y1 = par1(4);
    x2 = par2(3); y2 = par2(4);
    xn = par12(3);yn = par12(4);
    dist1 = abs(pdist2([x1 y1],[xn yn])) ;
    dist2 = abs(pdist2([x2 y2],[xn yn]));
    distance = abs(pdist2([x1 y1],[x2 y2]));
    if  ((distance > thd2) && (dist1 > thd1 || dist2 > thd1)) 
        dist = 0;
    else
        dist = 1;
    end
