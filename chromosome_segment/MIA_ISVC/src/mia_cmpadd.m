function ADD = mia_cmpadd(yx, ellipsparam)
% mia_estimateadd estimate "Average Distance Deviation (ADD)".

%   Synopsis
%       ADD = mia_estimateadd(yx, ellipsparam)
%   Description
%       measures the discrepancy between the fitted curve and the candidate contour
%       points. The lower value of ADD indicates higher goodness of fit

%   Inputs 
%         - contour      segment coordinates
%         - ellipsparam   ellipse parameters [r1 r2 cx cy theta]

%   Outputs
%         - ADD  the ADD value.
%         
%   Authors
%          Sahar Zafari <sahar.zafari(at)lut(dot)fi>
%
%   Changes
%       14/01/2016  First Edition

    nmyx = size(yx,2);
    if nmyx > 5 && ~isempty(ellipsparam)
        a = max(ellipsparam(1),ellipsparam(2));
        b = min(ellipsparam(1),ellipsparam(2));
        if (a == ellipsparam(1))
            theta = -ellipsparam(5);
        else
            theta = -(ellipsparam(5) + pi/2);
        end
        xy_eo  = [ellipsparam(3) ellipsparam(4)]'; 
        R = [ [ cos(theta) sin(theta)]', [-sin(theta) cos(theta)]'];
        xy = [yx(2,:);yx(1,:)];
        xypr  = R*( xy - repmat(xy_eo,1,nmyx) );
        D = sum(xypr.^2 ./ repmat([a^2 b^2]',1,nmyx)).^0.5;
        ADD = mean(abs(((sum(xypr.^2)).^0.5) .* (1 - 1./D)) ); 
    end
end
