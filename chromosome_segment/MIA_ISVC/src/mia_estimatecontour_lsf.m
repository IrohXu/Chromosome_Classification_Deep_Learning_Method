function stats =  mia_estimatecontour_lsf(I,contourevidence,vis)
%  mia_estimatecontour_lsf esitmate the objects boundaries by ellipse fitting.
%   Synopsis
%        result =  mia_lsf(I,contourevidence,vis)
%   Description
%        % the contour estimations is addressed through the classical ellipse
%          fitting problem by which the partially observed objects
%          are modelled in the form of ellipse-shape objects.
%          The goodness of the ellipse fitting is modeled
%          as the sum of squared algebraic distances of every point
%          involved.

%   Inputs 
%        - I                    grayscale or binar image
%        - contourevidence      contour evidences
%        - vis                  0 or 1, to visualize the result
%   Outputs
%         - result   boundary coordinates of ellipse
%         
%   Authors
%          Sahar Zafari <sahar.zafari(at)lut(dot)fi>
%
%
%   Changes
%       14/01/2016  First Edition


    stats = {};
    j = 0;
    for i=1:length(contourevidence)
        xe_i = contourevidence{i}(:,2);
        ye_i = contourevidence{i}(:,1);               
        if length(xe_i) > 15
            a = mia_fitellip_lsf(xe_i,ye_i); % elipse parameters
            v = mia_solveellipse_lsf(a);
            if ~isempty(a) && ~any(isnan(a))
                [X,Y]=mia_drawellip_lsf(v); % draw elipse
            end
            if (X~=0) 
                j = j + 1;
                stats{j} = [X' Y'];
            end
        end
    end
    if vis == 1
       fh = figure;
       imshow(I); hold on
       for i=1:length(stats)
         figure(fh); hold on;
         plot(stats{i}(:,1),stats{i}(:,2), 'color',[0.8 0.5 0.5],'LineWidth', 2);hold off;
       end
       title('Contour Estimation');
     end



end
    


