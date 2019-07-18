function [segments,centers] = mia_segmentcurve_concave(I,curve,idx,vis)
% mia_segmentcurve_concave split the curve into contour segments.

%   Synopsis
%       [segments,centers] = mia_segmentcurve_concave(imgbw,curve,idx,vis)
%   Description
%        Returns the contour segments and their centroids.The obtained
%        concave points are used to split the contours into contour 
%        segments contour segmentation.
%   Inputs 
%         - I       binary image.
%         - idx     cell array containin the index of detected corener in the input image.    
%         - curve   cell array contating the coordiantes of curve in the
%                   image, Matrix n-by-2 representing coordiantes of curve

%       
%        
%   Outputs
%        - segments    cell array contating the contour segments
%        - centers    Matrix n-by-2 representing centroids of the contour segments
%                           
%   Authors
%          Sahar Zafari <sahar.zafari(at)lut(dot)fi>
%
%   Changes
%       14/01/2016  First Edition


    segment = cell(1,size(idx,2));
    for j=1:size(idx,2) % iterate over connected components
        keep = [];
        seg =cell(1,length(idx{1,j}));
        if isempty(idx{1,j}) % curve without extermum 
             seg{1} = curve{j};
        end
        for i=1:length(idx{1,j}) %loop over concave points
            idxj =idx{1,j};
             if i==1 && idxj(1)~=1
                keep = curve{j}(1:idxj,:); % keep first segments
            end

            if i==length(idx{1,j})
                lastseg = curve{j}(idxj(i):end,:);
                seg{i} = [keep;lastseg];
            else
                seg{i} = curve{j}(idxj(i):idxj(i+1),:);

            end

        end
        segment{1,j} = seg;

    end
    % save all segments and center of mass
    k = 0;
    for i=1:length(segment) %length(segment)
      for ij=1:length(segment{i})
         k=k+1;
         segments{k} =segment{i}{ij};
         centers(k,:) = mean(segment{i}{ij});
      end
    end
    if vis == 1
        mia_visseg(I,segments);
        title('Segmented Contours by Concave points');
        fprintf('press any key to start segment grouping....\n');
        pause;
    end
end
