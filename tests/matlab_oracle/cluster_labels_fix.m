function [labels,nlabels,stack,isink,DBG]=cluster_labels_fix(ptCloud,param,indNeighbors,labels,nlabels,stack,ndon,isink,surface,normals)
% INSTRUMENTED COPY of Utils/cluster_labels.m (production detector untouched).
% Byte-faithful to the algorithm; adds a 5th output DBG capturing the internal
% matrices (Dist, Nneigh, Aangle, Mmerge, dbscan idx, indborder) so the Python
% stage-parity tests can check the merge stage against the MATLAB oracle.

% ---- Inter-distance between isinks associated to each label
D1=pdist2(ptCloud.Location(isink,:),ptCloud.Location(isink,:));
A=zeros(1,nlabels);for k=1:nlabels;A(k) = sum(surface(stack{k}));end;radius=sqrt(A./pi);
D2=zeros(nlabels,nlabels);D2=D2+radius+radius';
Dist=zeros(nlabels,nlabels);ind=find(param.radfactor.*D2>D1);Dist(ind)=1;Dist=Dist-eye(size(Dist));

% ---- Determine if labels are neighbours
Nneigh=zeros(nlabels,nlabels);for k=1:nlabels;ind=unique(labels(indNeighbors(stack{k},:)));Nneigh(k,ind)=1;end

% ---- normals at the border of labels
temp=param.nnptCloud-sum((labels(indNeighbors)==labels),2);indborder=find(temp>=param.nnptCloud/4 & ndon==0);
A=zeros(nlabels,nlabels);N=zeros(nlabels,nlabels);
for k=1:numel(indborder)
    i=indborder(k); j=indNeighbors(i,:);
    P1=repmat(normals(i,:),param.nnptCloud,1);P2=normals(j,:);
    A(labels(i),labels(j))=A(labels(i),labels(j))+anglerot2vecmat(P1,P2);
    N(labels(i),labels(j))=N(labels(i),labels(j))+1;
end
Aangle=A./N;

% ---- Merge grains
Mmerge=zeros(nlabels,nlabels);Mmerge(Dist<1 | Nneigh<1 | Aangle>param.maxangle1)=Inf;
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

DBG=struct('Dist',Dist,'Nneigh',Nneigh,'Aangle',Aangle,'Mmerge',Mmerge, ...
           'dbscan_idx',idx,'indborder',indborder,'nstack',nstack);
end
