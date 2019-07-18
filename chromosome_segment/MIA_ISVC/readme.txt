Microscopy Image Analysis (MIA) toolbox for segmentation of overlapping convex objects in microscopy images.
% Version 1.0 15-Jan-2016
The toolbox contains the following folders:
src:


	%   mia_cmpadd                   - Compute the discrepancy between the fitted curve and the contour points. 
	%   mia_cmpconcavepoint_css 	 - Performs concave point extraction.
	%   mia_contourevidence 	 - performs contour evidenc extraction step of the method.
	%   mia_cmpdistance  		 - computes the distance betwwen fitted ellipses.
	%   mia_groupsegments 		 - Performs segment grouping sub-step of the method.
	%   mia_removeconvexcorner       - Remove the convex corner.
	%   mia_segmentcurve_concave     - Split the curve into contour segments.
	%   mia_estimatecontour		 - Esitmate the objects boundaries by ellipse fitting.

	% Demos
	%   readparam                   - Read the method parameters.
	%   config                      - Add necessary path.
	%   demo                        - Demonstrate all the method steps in one example image.
	%   mia_particles_segmentation -performs segmentation by using the concave points.
lib: 
	% modified/available matlab codes or toolbox
	%   mia_curve_tangent 		  -Estimate the curve tangent.
	%   mia_get_corner  		  -Compute the corner in the image.
	%   mia_extract_curve 		 - Extract the curve in the image.
	%   mia_beresenham               - Creat a line beetween two point based on bresenham algorithm.
	%   mia_cmpparam      		 - Returns the paramters of ellipses fitted to the evidences.
	%   mia_fitellip_lsf     	 - Returns the 6 parameter vector of the algebraic circle fit
	%                        	   to a(1)x^2 + a(2)xy + a(3)y^2 + a(4)x + a(5)y + a(6) = 0.
	%   mia_drawellip_lsf      	 - Draw the ellipse by 100 points.
	%   mia_solveellipse_lsf         - Returns the ellipse parameters [r1 r2 cx cy theta].

img:
      % the input images


% References:

%   [1] Zafari, S.; Eerola, T.; Sampo, J.; Kalviainen, H.; Haario, H.,
%   "Segmentation of Partially Overlapping Nanoparticles Using Concave Points," 
%   in International Symposium on Visual Computing (ISVC 2015), 2015.


% Author(s):

%    Sahar Zafari <sahar.zafari@lut.fi>

%  Please, if you find any bugs contact the authors.


