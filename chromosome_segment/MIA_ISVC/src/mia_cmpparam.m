function param = mia_cmpparam(yx)
% mia_cmpparam estimate the paramtes of ellipse fitted to the segment.
%   Synopsis
%       param = mia_estimateparam(yx)
%   Description
%         Returns the paramters of ellipses fitted to the segemnts
%         with least square fitting method.
%         contour segments.
%   Inputs 
%          - yx   segments coordinates
%   Outputs
%         - param  ellipse parameters
%         
%   Authors
%          Sahar Zafari <sahar.zafari(at)lut(dot)fi>
%
%   Changes
%       14/01/2016  First Edition


    param = [];
    nmyx = size(yx,2);
    if nmyx>10
       a = mia_fitellip_lsf(yx(2,:)',yx(1,:)');
       if isreal(a)
          param = mia_solveellipse_lsf(a);
       end
     end
