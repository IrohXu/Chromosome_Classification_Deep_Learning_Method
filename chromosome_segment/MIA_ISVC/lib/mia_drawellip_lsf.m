function [X,Y]=mia_drawellip_lsf(v)
% mia_drawellip_lsf draw elipse.
%   Synopsis
%        [X,Y]=mia_drawellip_lsf(a)
%   Description
%        drwa the ellipse by 100 points.

%   Inputs 
%        - v   param  ellipse parameters [r1 r2 cx cy theta]
%   Outputs
%         - x y   boundary coordinates of ellipse
%         
%   Authors
%	code is downloaded and modfied from http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/PILU1/demo.html
%	M. Pilu, A. Fitzgibbon, R.Fisher ``Ellipse-specific Direct least-square Fitting '' , 
%	IEEE International Conference on Image Processing, Lausanne, September 1996. 
%
%
%   Changes
%       14/01/2016  First Edition

  
   if isempty(v)
       X =0; Y=0; v=0;
   else
   
       % draw ellipse with N points   
       N = 100;
       dx = 2*pi/N;
       theta = v(5);
       R = [ [ cos(theta) sin(theta)]', [-sin(theta) cos(theta)]'];
       for i = 1:N
            ang = i*dx;
            x = v(1)*cos(ang);
            y = v(2)*sin(ang);
            d1 = R*[x y]';
            X(i) = d1(1) + v(3);
            Y(i) = d1(2) + v(4);
       end
   end
