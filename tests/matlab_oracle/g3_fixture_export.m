% g3_fixture_export.m -- Phase 0 MATLAB oracle for the g3point_python revival.
%
% Runs the byte-faithful G3Point pipeline (same sequence as g3_batch.m / g3_headless.m)
% on each fixture tile and DUMPS EVERY STAGE INTERMEDIATE to <stem>_fixture.mat, so the
% Python port can be tested stage-by-stage against the MATLAB reference (not just on the
% final GSD). Uses instrumented COPIES cluster_labels_fix / clean_labels_fix (production
% Utils/ untouched) to capture the internal merge matrices + dbscan idx. Records MATLAB
% version, params, and point counts per stage in a meta struct.
%
% env:
%   G3_FIX_TILEDIR  dir with tile .ply files
%   G3_FIX_LIST     text file, one tile .ply basename per line
%   G3_FIX_OUT      output dir for <stem>_fixture.mat
% Run from G3Point/G3point (so Utils/ is on path); this file's dir must also be on path
% for the *_fix copies. Uses defineparameters manual defaults (no param.csv).

addpath('Utils','-end'); addpath('Utils/quadfit','-end'); addpath('Utils/geom3d/geom3d/','-end');
addpath(fileparts(mfilename('fullpath')),'-end');      % scripts/ (holds nothing) -- ensure fixtures/ on path:
addpath(fullfile(fileparts(mfilename('fullpath')),'fixtures'),'-end');

tiledir = getenv('G3_FIX_TILEDIR'); if isempty(tiledir); error('G3_FIX_TILEDIR unset'); end
if tiledir(end) ~= filesep; tiledir = [tiledir filesep]; end
listfile = getenv('G3_FIX_LIST'); if isempty(listfile); error('G3_FIX_LIST unset'); end
outdir = getenv('G3_FIX_OUT'); if isempty(outdir); outdir = tiledir; end
if ~exist(outdir,'dir'); mkdir(outdir); end
if outdir(end) ~= filesep; outdir = [outdir filesep]; end

fid = fopen(listfile); C = textscan(fid, '%s'); fclose(fid);
tiles = C{1};
fprintf('=== G3 FIXTURE EXPORT: %d tiles ===\n', numel(tiles));
mlver = version;

for it = 1:numel(tiles)
    plyname = tiles{it};
    stem = plyname(1:end-4);
    t0 = tic;
    try
        param = struct(); param.ptCloudname = plyname; param.ptCloudpathname = tiledir;

        % --- load (min-shift + removeInvalidPoints)
        raw = pcread([tiledir plyname]); n_raw = size(raw.Location,1);
        [ptCloud, param] = loadptCloud(param);
        xyz_loaded = ptCloud.Location; n_loaded = size(xyz_loaded,1);

        param = defineparameters(ptCloud, param);
        param.iplot=0; param.saveplot=0; param.gridbynumber=0; param.savegrain=0;

        % --- denoise (capture inlier indices into the loaded cloud)
        if param.denoise==1
            [ptCloud, inlierIdx] = pcdenoise(ptCloud);
        else
            inlierIdx = (1:n_loaded)';
        end
        xyz_denoised = ptCloud.Location; n_denoised = size(xyz_denoised,1);

        % --- rotate + detrend -> ptCloudRot (used only for segmentation)
        [A,B,Cc,~,~]=fitplan(ptCloud.Location); Normal=[-A -B 1];
        [Normal]=adjustnormals3d(0,0,0,Normal,[0 0 1e32]);
        R=vec2rot(Normal,[0 0 1],'Rik'); meanptCloud=mean(ptCloud.Location);
        xyzPoints=(R*(ptCloud.Location-meanptCloud)')'+meanptCloud;
        ptCloudRot=pointCloud(xyzPoints);
        x=ptCloudRot.Location(:,1);y=ptCloudRot.Location(:,2);z=ptCloudRot.Location(:,3);
        Ac=[ones(size(x)) x y x.^2 x.*y y.^2]\z;
        z=z-(Ac(1)+Ac(2).*x+Ac(3).*y+Ac(4).*x.^2+Ac(5).*x.*y+Ac(6).*y.^2);
        ptCloudRot=pointCloud([x y z]);
        xyz_detrended = ptCloudRot.Location;

        % --- kNN + surfaces (on original/denoised frame)
        [indNeighbors,D]=knnsearch(ptCloud.Location,ptCloud.Location,'K',param.nnptCloud+1);
        indNeighbors=indNeighbors(:,2:end); D=D(:,2:end);
        surface=pi.*min(D,[],2).^2;

        % --- normals (on original frame)
        normals=pcnormals(ptCloud,param.nnptCloud);
        [normals]=adjustnormals3d(ptCloud.Location(:,1),ptCloud.Location(:,2),ptCloud.Location(:,3), ...
                                  normals,[mean(ptCloud.Location(:,1)),mean(ptCloud.Location(:,2)),10000]);

        % --- initial segmentation (on detrended frame)
        [labels_seg,nlabels_seg,labelsnpoint,stack,nstack,ndon,isink]=segment_labels(ptCloudRot,param,indNeighbors);

        % --- cluster (instrumented)
        [labels_cl,nlabels_cl,stack,isink,DBG_cluster]=cluster_labels_fix(ptCloud,param,indNeighbors,labels_seg,nlabels_seg,stack,ndon,isink,surface,normals);

        % --- clean (instrumented)
        [labels_clean,nlabels_clean,stack,isink,DBG_clean]=clean_labels_fix(ptCloud,param,indNeighbors,labels_cl,nlabels_cl,stack,ndon,isink,surface,normals);

        % --- ellipsoid fit + per-grain quality (on original frame Pebble coords)
        clear Pebble
        for i=1:nlabels_clean; ind=find(labels_clean==i); Pebble(i).Location=ptCloud.Location(ind,:); Pebble(i).ind=ind; Pebble(i).surface=surface(ind); end
        [Ellipsoidm]=fitellipsoidtograins(Pebble,param,nlabels_clean);
        fitok=[Ellipsoidm.fitok]; aqok=[Ellipsoidm.Aqualityok];
        acover=nan(1,nlabels_clean); radii=nan(3,nlabels_clean); centers=nan(3,nlabels_clean); Rmats=nan(9,nlabels_clean);
        for i=1:nlabels_clean
            if isfield(Ellipsoidm(i),'Acover') && ~isempty(Ellipsoidm(i).Acover); acover(i)=Ellipsoidm(i).Acover; end
            if isfield(Ellipsoidm(i),'r') && ~isempty(Ellipsoidm(i).r); radii(:,i)=Ellipsoidm(i).r(:); end
            if isfield(Ellipsoidm(i),'c') && ~isempty(Ellipsoidm(i).c); centers(:,i)=Ellipsoidm(i).c(:); end
            if isfield(Ellipsoidm(i),'R') && ~isempty(Ellipsoidm(i).R); Rmats(:,i)=Ellipsoidm(i).R(:); end
        end
        [granulo]=grainsizedistribution(Ellipsoidm);

        meta=struct('tile',stem,'matlab_version',mlver,'n_raw',n_raw,'n_loaded',n_loaded, ...
                    'n_denoised',n_denoised,'nlabels_seg',nlabels_seg,'nlabels_cluster',nlabels_cl, ...
                    'nlabels_clean',nlabels_clean,'param',param);

        outmat=[outdir stem '_fixture.mat'];
        save(outmat,'meta','xyz_loaded','inlierIdx','xyz_denoised','xyz_detrended', ...
             'indNeighbors','D','surface','normals','labels_seg','isink','ndon', ...
             'labels_cl','DBG_cluster','labels_clean','DBG_clean', ...
             'fitok','aqok','acover','radii','centers','Rmats','granulo','-v7');
        fprintf('[%d/%d] %s: seg=%d clust=%d clean=%d grains, denoise %d->%d (%.1fs) SAVED\n', ...
                it,numel(tiles),stem,nlabels_seg,nlabels_cl,nlabels_clean,n_loaded,n_denoised,toc(t0));
    catch ME
        fprintf('[%d/%d] %s: FIXTURE FAILED (%.1fs): %s\n', it,numel(tiles),stem,toc(t0),ME.message);
    end
end
fprintf('=== FIXTURE EXPORT DONE ===\n');
