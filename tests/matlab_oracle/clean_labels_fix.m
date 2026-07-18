function [labels,nlabels,stack,isink,DBG]=clean_labels_fix(ptCloud,param,indNeighbors,labels,nlabels,stack,ndon,isink,surface,normals)
% INSTRUMENTED COPY of Utils/clean_labels.m (production detector untouched).
% Byte-faithful; adds DBG capturing the clean-stage merge matrices, the dbscan
% idx, and the two keep-masks (small-label and flatness) so the Python
% stage-parity tests can check the clean stage against the MATLAB oracle.

% ---- normals at the border of labels
temp=param.nnptCloud-sum((labels(indNeighbors)==labels),2);
indborder=find(temp>=param.nnptCloud/4 & ndon==0);
A=zeros(nlabels,nlabels);N=zeros(nlabels,nlabels);
for k=1:numel(indborder)
    i=indborder(k); j=indNeighbors(i,:);
    P1=repmat(normals(i,:),param.nnptCloud,1);P2=normals(j,:);
    A(labels(i),labels(j))=A(labels(i),labels(j))+anglerot2vecmat(P1,P2);
    N(labels(i),labels(j))=N(labels(i),labels(j))+1;
end
Aangle=A./N;Aangle(N==0)=0;

% ---- Merge grains
Mmerge=zeros(nlabels,nlabels);Mmerge(Aangle>param.maxangle2 | Aangle==0)=Inf;
[idx, ~] = dbscan(Mmerge,1,1,'Distance','precomputed');
newlabels=zeros(size(labels));newstack=cell(numel(unique(idx)),1);
for i=1:numel(unique(idx))
    ind=find(idx==i);
    for j=1:numel(ind)
        newlabels(stack{ind(j)})=i;
        newstack{i}=[newstack{i} stack{ind(j)}];
    end
end
labels=newlabels;nlabels=max(labels);stack=newstack;nstack=cellfun(@numel, stack);
clear isink;for i=1:nlabels;[~,temp]=max(ptCloud.Location(stack{i},3));isink(i)=stack{i}(temp);end

% ---- Remove small labels
nstack_before_small=nstack;
clear newstack newisink newlabels
indtokeep=find(nstack>=param.minnpoint);newlabels=0*labels;
keep_small=indtokeep;
for k=1:numel(indtokeep)
    newstack{k}=stack{indtokeep(k)};
    newisink(k)=isink(indtokeep(k));
    newlabels(newstack{k})=k;
end
stack=newstack;isink=newisink;labels=newlabels;nlabels=max(labels);nstack=cellfun(@numel, stack);

% ---- Remove flattish labels
clear newstack newisink newlabels
r=zeros(nlabels,3);for k=1:nlabels;r(k,1:3)=svd(ptCloud.Location(stack{k},1:3)-mean(ptCloud.Location(stack{k},1:3)));end
indtokeep=find(r(:,3)./r(:,1)>param.minflatness | r(:,2)./r(:,1)>2.*param.minflatness);newlabels=0*labels;
keep_flat=indtokeep;svd_ratios=r;
for k=1:numel(indtokeep)
    newstack{k}=stack{indtokeep(k)};
    newisink(k)=isink(indtokeep(k));
    newlabels(newstack{k})=k;
end
stack=newstack;isink=newisink;labels=newlabels;nlabels=max(labels);nstack=cellfun(@numel, stack);

labels(labels==0)=NaN;

DBG=struct('Aangle',Aangle,'Mmerge',Mmerge,'dbscan_idx',idx, ...
           'indborder',indborder,'nstack_before_small',nstack_before_small, ...
           'keep_small',keep_small,'keep_flat',keep_flat,'svd_ratios',svd_ratios);
end
