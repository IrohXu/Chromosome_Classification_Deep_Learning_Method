function a = mia_fitellip_lsf(X,Y)
% mia_fitellip_lsf estimate the paramtes of ellipse fitted to the segment
%   Synopsis
%       param = mia_estimateparam(yx)
%   Description
%        fitellip gives the 6 parameter vector of the algebraic circle fit
%        to a(1)x^2 + a(2)xy + a(3)y^2 + a(4)x + a(5)y + a(6) = 0.
%   Inputs 
%          - X & Y are lists of point coordinates and must be column vectors.
%   Outputs
%         - param  ellipse parameters
%         
%   Authors
%	code is downloaded and modfied from http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/PILU1/demo.html
%	M. Pilu, A. Fitzgibbon, R.Fisher ``Ellipse-specific Direct least-square Fitting '' , 
%	IEEE International Conference on Image Processing, Lausanne, September 1996. 
%
%   Changes
%       14/01/2016  First Edition

% 

   % normalize data
   mx = mean(X);
   my = mean(Y);
   sx = (max(X)-min(X))/2;
   sy = (max(Y)-min(Y))/2;
   x = (X-mx)/sx;
   y = (Y-my)/sy;
   
   % Build design matrix
   D = [ x.*x  x.*y  y.*y  x  y  ones(size(x)) ];

   % Build scatter matrix
   S = D'*D;

   % Build 6x6 constraint matrix
   C(6,6) = 0; C(1,3) = -2; C(2,2) = 1; C(3,1) = -2;
   
   if any(isnan(S))
      a = [];
      return;
   end
   % Solve eigensystem
   [gevec, geval] = eig(S,C);

   % Find the negative eigenvalue
   [~, NegC] = find(geval < 0 & ~isinf(geval));
   
   % Extract eigenvector corresponding to positive eigenvalue
   A = gevec(:,NegC);
   
   if numel(A)==0
       a = [];
       return

   end

       % unnormalize
   a = [A(1)*sy*sy,   ...
        A(2)*sx*sy,   ...
        A(3)*sx*sx,   ...
        -2*A(1)*sy*sy*mx - A(2)*sx*sy*my + A(4)*sx*sy*sy,   ...
        -A(2)*sx*sy*mx - 2*A(3)*sx*sx*my + A(5)*sx*sx*sy,   ...
        A(1)*sy*sy*mx*mx + A(2)*sx*sy*mx*my + A(3)*sx*sx*my*my   ...
        - A(4)*sx*sy*sy*mx - A(5)*sx*sx*sy*my   ...
        + A(6)*sx*sx*sy*sy   ...
        ]';
   
end

