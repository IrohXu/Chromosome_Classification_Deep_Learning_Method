function mia_visseg(I,segment)
    col_seg = linspecer(length(segment)) ;
    figure(); imshow(I),hold on
    for i=1:length(segment)
       scatter(segment{i}(:,2),segment{i}(:,1),[],'MarkerEdgeColor',col_seg(i,:,:),'MarkerFaceColor',col_seg(i,:,:))
%        pause
    end
end

