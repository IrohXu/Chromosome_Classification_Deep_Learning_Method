function idxpp = mia_removeconvexcorner(idx,curve,mask,k,vis)
% mia_removeconvexcorner remove the convex corner.

%   Synopsis
%       [xc,yc,idxpp] = mia_removeconvexcorner(idx,curve,mask)
%   Description
%        Returns the concave points in objects boundary.The corner 
%        point pi is qualified as concave if the line
%        connecting p i−k to p i+k does not reside inside 
%        contour segmentation and segment grouping
%   Inputs 
%         - idx      index of detected corener in the input image.    
%         - curve   objects boundareis
%         - mask    binar mask of image
%         - k       kth adjacent contour points
%   Outputs
%        - idxpp  index of detected concave in the input image.
%        - vis    0 or 1, for visualizing the dtected concave points.
%                           
%   Authors
%          Sahar Zafari <sahar.zafari(at)lut(dot)fi>
%
%   Changes
%       14/01/2016  First Edition

     se = strel('disk',1); 
     mask = imdilate(mask,se);
    [nmrows, nmcols] = size(mask);
    l = 0;
    for i = 1:size(idx,2)
        if length(idx{i})== 0
            idxpp{i}=[];
        end
         for j = 1:length(idx{i})
            idxi =idx{i}(j);
            x = curve{i}(idxi,2);
            y = curve{i}(idxi,1);
            xyp = curve{i}(max(1,mod(idxi+k,size(curve{i},1))),:);
            xym = curve{i}(max(1,mod(end+(idxi-k),size(curve{i},1))),:);
            [psubx,psuby]=mia_bresenham(xyp(2),xyp(1),xym(2),xym(1));
            pidx = sub2ind([nmrows, nmcols], psuby, psubx);
            th = 0.90; 
            ratfg = sum(mask(pidx))/length(pidx);
            res = ratfg > th;
          if res
                idxpp{i}(j) = 0;
              continue;
          else
              l = l+1;
              xc(l) = x;
              yc(l) = y;
              idxpp{i}(j) = idxi;
         end

        end
    end
    for i=1:length(idxpp)
      if  ~isempty(idxpp{i})
       idxpp{i}(idxpp{i}==0)=[];
      end
    end
     if vis == 1
        figure(1);
        imshow(mask); hold on
        plot(xc,yc,'sg','markerfacecolor','g'); hold on
        title('Detected Concave points')
        fprintf('press any key to start contour segmentation...\n')
        pause;
    end
    end
