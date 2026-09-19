function prf_analyzeprf_fit(matfile, outdir, negate, nworkers, chunksize)
% prf_analyzeprf_fit  Run analyzePRF on exported pooled pseudo-runs, in chunks.
%
%   prf_analyzeprf_fit(matfile, outdir, negate, nworkers, chunksize)
%
% Step 2 of the analyzePRF-backed pRF fit (workbench prf-retinotopy, DECIDED
% 2026-09-16: handle the data EXACTLY as analyzePRF does). The call is NSD's
% own, verbatim from cvnlab/nsddatapaper main/analysis_prf.m:
%
%   analyzePRF(stimulus, data, tr, struct('seedmode',2,'maxiter',100,'display','off'))
%
% i.e. super-grid seeds, 100 iterations, free exponent, analyzePRF's default
% HRF (getcanonicalhrf) and default polynomial degree, every voxel handed in,
% nothing gated. NSD chunked at 100k voxels only for memory; chunking here is
% for RESUME: each chunk lands in its own file and an existing file is
% skipped, so a re-queued job continues where the last one stopped.
%
% Inputs
%   matfile   from prf_analyzeprf_export.py: stimulus (1 x runs cell of
%             res x res x T), data (1 x runs cell of vox x T), tr
%   outdir    chunk-NNNN.mat files + fit-manifest.json land here
%   negate    true/1 = fit -data (negative-pRF arm; the negprf suffix)
%   nworkers  parpool size (the job's cpus-per-task)
%   chunksize voxels per chunk (default 50000)
%
% Paths to the toolboxes come from the environment, set by the sbatch from
% config/base.toml: ANALYZEPRF_DIR and KNKUTILS_DIR.

matfile = char(matfile); outdir = char(outdir);   % string inputs would make [x '.tmp'] a string array
if nargin < 5 || isempty(chunksize), chunksize = 50000; end
if nargin < 4 || isempty(nworkers), nworkers = feature('numcores'); end
if ischar(negate) || isstring(negate), negate = str2double(negate); end
if ischar(nworkers) || isstring(nworkers), nworkers = str2double(nworkers); end
if ischar(chunksize) || isstring(chunksize), chunksize = str2double(chunksize); end
negate = logical(negate);

aprf = getenv('ANALYZEPRF_DIR'); knk = getenv('KNKUTILS_DIR');
if isempty(aprf) || isempty(knk)
    error('prf_analyzeprf_fit:paths', ...
        'ANALYZEPRF_DIR and KNKUTILS_DIR must be set (config/base.toml [paths] via load_config.sh)');
end
if ~exist(fullfile(aprf, 'analyzePRF.m'), 'file')
    error('prf_analyzeprf_fit:paths', 'no analyzePRF.m under %s', aprf);
end
addpath(genpath(aprf)); addpath(genpath(knk));

t_all = tic;
S = load(matfile);
stimulus = cellfun(@double, S.stimulus, 'UniformOutput', false);
data = cellfun(@double, S.data, 'UniformOutput', false);
tr = double(S.tr);
nvox = size(data{1}, 1);
if negate
    data = cellfun(@(x) -x, data, 'UniformOutput', false);
end
fprintf('prf_analyzeprf_fit: %d runs, %d voxels x %d TRs, TR %.3f, negate %d, chunk %d, workers %d\n', ...
    numel(data), nvox, size(data{1}, 2), tr, negate, chunksize, nworkers);

if ~exist(outdir, 'dir'), mkdir(outdir); end

% NSD analysis_prf.m call, verbatim. Do not add options here: any change is a
% deviation from "handled exactly as analyzePRF" and belongs in the workbench
% first.
opt = struct('seedmode', 2, 'maxiter', 100, 'display', 'off');

% One parpool per job, with its storage on node-local scratch so concurrent
% array tasks do not share a JobStorageLocation.
jobtag = getenv('SLURM_JOB_ID'); if isempty(jobtag), jobtag = 'local'; end
c = parcluster('Processes');
c.JobStorageLocation = fullfile(tempdir, ['prf_analyzeprf_' jobtag]);
if ~exist(c.JobStorageLocation, 'dir'), mkdir(c.JobStorageLocation); end
c.NumWorkers = max(nworkers, c.NumWorkers);
delete(gcp('nocreate'));
parpool(c, nworkers);

nchunks = ceil(nvox / chunksize);
done = 0;
for k = 1:nchunks
    out = fullfile(outdir, sprintf('chunk-%04d.mat', k));
    if exist(out, 'file')
        fprintf('  chunk %d/%d exists, skipping\n', k, nchunks);
        done = done + 1;
        continue;
    end
    vxs = (k - 1) * chunksize + 1 : min(k * chunksize, nvox);
    dchunk = cellfun(@(x) x(vxs, :), data, 'UniformOutput', false);
    t_chunk = tic;
    results = analyzePRF(stimulus, dchunk, tr, opt);
    elapsed = toc(t_chunk);
    fprintf('  chunk %d/%d: %d voxels in %.1f min (%.2f s/voxel/worker)\n', ...
        k, nchunks, numel(vxs), elapsed / 60, elapsed * nworkers / numel(vxs));
    % analyzePRF's own outputs, one row per voxel of this chunk; pixel units,
    % converted to degrees by prf_analyzeprf_assemble.py. numiters is a cell
    % (one entry per seed) -- kept as the total over seeds.
    chunk = struct();
    chunk.vxs = vxs(:);
    chunk.ang = results.ang(:);
    chunk.ecc = results.ecc(:);
    chunk.expt = results.expt(:);
    chunk.rfsize = results.rfsize(:);
    chunk.R2 = results.R2(:);
    chunk.gain = results.gain(:);
    chunk.params = reshape(results.params, 5, [])';   % voxels x 5 [row col sigma gain expt]
    chunk.numiters = cellfun(@(x) sum(x(:)), results.numiters);
    chunk.elapsed_s = elapsed;
    tmp = [out '.tmp'];
    save(tmp, '-struct', 'chunk', '-v7');
    movefile(tmp, out);
    done = done + 1;
end

% What was actually used, from analyzePRF's own record of its options.
manifest = struct();
manifest.call = 'analyzePRF(stimulus, data, tr, struct(''seedmode'',2,''maxiter'',100,''display'',''off''))';
manifest.seedmode = opt.seedmode;
manifest.maxiter = opt.maxiter;
manifest.hrf = results.options.hrf(:)';
manifest.maxpolydeg = results.options.maxpolydeg;
manifest.exptlowerbound = results.options.exptlowerbound;
manifest.typicalgain = results.options.typicalgain;
manifest.negate = negate;
manifest.n_vox = nvox;
manifest.n_runs = numel(data);
manifest.chunksize = chunksize;
manifest.n_chunks = nchunks;
manifest.matlab = version;
manifest.nworkers = nworkers;
manifest.elapsed_min = toc(t_all) / 60;
fid = fopen(fullfile(outdir, 'fit-manifest.json'), 'w');
fprintf(fid, '%s\n', jsonencode(manifest, 'PrettyPrint', true));
fclose(fid);
fprintf('prf_analyzeprf_fit: %d/%d chunks on disk, %.1f min total\n', done, nchunks, toc(t_all) / 60);
delete(gcp('nocreate'));
end
