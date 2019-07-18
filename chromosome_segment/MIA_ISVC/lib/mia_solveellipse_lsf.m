function v = mia_solveellipse_lsf(a)
% mia_solveellipse_lsf returning [r1 r2 cx cy theta]
%   Synopsis
%       v = mia_solveellipse_lsf(a)
%   Description
%        % Given an ellipse in the form:
%       a(1)x^2 + a(2)xy + a(3)y^2 + a(4)x + a(5)y + a(6) = 0
%       finds the standard form:
%       ((x-cx)/r1)^2 + ((y-cy)/r2)^2 = 1
%       returning [r1 r2 cx cy theta]
%   Inputs 
%          - a    the 6 parameter vector of the algebraic circle fit
%        to a(1)x^2 + a(2)xy + a(3)y^2 + a(4)x + a(5)y + a(6) = 0.
%   Outputs
%         - param  ellipse parameters [r1 r2 cx cy theta]
%         
%   Authors
%	code is downloaded and modfied from http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/PILU1/demo.html
%	M. Pilu, A. Fitzgibbon, R.Fisher ``Ellipse-specific Direct least-square Fitting '' , 
%	IEEE International Conference on Image Processing, Lausanne, September 1996. 
%
%
%   Changes
%       14/01/2016  First Edition


    if ~isempty(a)
        % get ellipse orientation
        theta = atan2(a(2),a(1)-a(3))/2;

        % get scaled major/minor axes
        ct = cos(theta);
        st = sin(theta);
        ap = a(1)*ct*ct + a(2)*ct*st + a(3)*st*st;
        cp = a(1)*st*st - a(2)*ct*st + a(3)*ct*ct;

        % get translations
        T = [[a(1) a(2)/2]' [a(2)/2 a(3)]'];
        if det(T)==0
            v = [];
            return;
        end
        t = -inv(2*T)*[a(4) a(5)]';
        cx = t(1);
        cy = t(2);

        % get scale factor
        val = t'*T*t;
        scale = 1 / (val- a(6));

        % get major/minor axis radii
        r1 = 1/sqrt(scale*ap);
        r2 = 1/sqrt(scale*cp);
        v = [r1 r2 cx cy theta]';
    else
        v = [];
    end

        
