function contourevidence = mia_groupsegments(I,segin,cntin,thdn,thd1,thd2,vis)
% mia_groupsegments performs segment grouping sub-step of the method.

%   Synopsis
%       contourevidence = mia_groupsegments(segin,cntin,thdn)
%   Description
%        Segment grouping iterates over each pair of
%        contour segment, examining if they can be combined.
%        To optimize the grouping process, a limited search space
%         is applied and the contour segment under grouping process 
%        is only examined with the neighbouring segments. 
%   Inputs 
%         - I           binary image
%         - segin      cell array contating the contour segments for grouping
%         - cntin      a Matrix n-by-2 representing centroids of the contour segments
%         - thdn       Euclidean distance between contour center points
%                       to define neighbouring segments 
%         - thd1       Euclidean distance between ellipse centroid of the 
%                      combined contour segments and ellipse fitted to each segment.
%         - thd2       Euclidean distance between between the centroids of ellipse
%                      fitted to each segment.

%   Outputs
%         - contourevidence    cell array contating the visile objects boundaries 

%   Authors
%          Sahar Zafari <sahar.zafari(at)lut(dot)fi>
%
%   Changes
%       14/01/2016  First Edition

    nmseg = length(segin);
    i = 0;
    idxdel = [];
    idxseg = 1:nmseg;
    while i< nmseg% iterate over connected components
       i = i + 1;
       if ismember(i,idxdel)
         continue
       end
        s1 = segin{i};
        idx = idxseg(~ismember(1:nmseg,idxdel));
        dnei = pdist2(cntin(i,:),cntin(idx,:));
        idxneiseg = idx(dnei<thdn & dnei>0);
        segnei = segin(idxneiseg);
        nmneiseg = length(idxneiseg);
        params1 = mia_cmpparam(s1');
        if numel(params1)==0 
            continue
        end
        for j=1:nmneiseg
            s2 = segnei{j};
            params2 = mia_cmpparam(s2');
            if numel(params2)==0 
                continue,
            end
            s12 = [s2;s1];
            params12 = mia_cmpparam(s12');
            if numel(params12)==0
                continue;
            end
            adds1 = mia_cmpadd(s1', params1); % add 1
            adds2 = mia_cmpadd(s2', params2); %add 2
            adds12 = mia_cmpadd(s12', params12); %add 12
            w1 = (length(s1)/(length(s1)+length(s2)));
            w2 = (length(s2)/(length(s1)+length(s2)));
            if length(s1) > length(s2)
                adds1 = w1*adds1;    
            else
                adds2 = w2*adds2; 
            end
            delp = mia_cmpdistance(params1,params2,params12,thd1,thd2);
            isgrouped =  (adds12 < adds1 || adds12 < adds2) && delp ; %grouping crietria
            if isgrouped
                segin{i} = s12;
                s1 = s12;
                idxdel = [idxdel,idxneiseg(j)];
                cntin(i,:) = mean(s12);
                break;
            end
        end
    end
    contourevidence = segin(~ismember(1:nmseg,idxdel));
    if vis == 1
        mia_visseg(I,contourevidence);
        title('Contour Evidences');
        fprintf('press any key to start contour estimation....\n')
        pause
    end
end



