function nordic_denoise(bold_in, out_dir)
% NORDIC_DENOISE  Apply magnitude-only NORDIC denoising to a BOLD NIfTI.
%
%   nordic_denoise(bold_in, out_dir)
%
%   bold_in  - full path to input *_bold.nii.gz
%   out_dir  - output directory (preserves original filename)
%
% Requires NORDIC_Raw on the MATLAB path.

if nargin < 2
    error('Usage: nordic_denoise(bold_in, out_dir)');
end

% Validate input
if ~isfile(bold_in)
    error('Input file not found: %s', bold_in);
end

if ~isfolder(out_dir)
    mkdir(out_dir);
end

% Derive output filename (same basename as input)
[~, fname, ext] = fileparts(bold_in);
% Handle .nii.gz double extension
if endsWith(fname, '.nii')
    fname_base = fname(1:end-4);
    ext = '.nii.gz';
else
    fname_base = fname;
end
% NIFTI_NORDIC concatenates ARG.DIROUT + fn_out + '.nii', so fn_out must
% be just the basename (no directory component).
fn_out = fname_base;

% Configure NORDIC for magnitude-only fMRI
ARG.DIROUT = [out_dir '/'];
ARG.magnitude_only = 1;
ARG.temporal_phase = 1;          % fMRI mode
ARG.kernel_size_PCA = [];        % use default (11:1 spatial:temporal ratio)
% Leave kernel_size_gfactor at default (empty). Setting it to [14 14 1]
% triggers a bug in NIFTI_NORDIC line 358 that indexes element 4.
ARG.save_add_info = 1;           % save noise estimate + degrees removed
ARG.write_gzipped_niftis = 1;    % output .nii.gz

fprintf('=== NORDIC Denoising ===\n');
fprintf('Input:  %s\n', bold_in);
fprintf('Output: %s\n', fn_out);
fprintf('Mode:   magnitude-only\n');
tic;

% Run NORDIC (fn_phase_in = [] for magnitude-only)
NIFTI_NORDIC(bold_in, [], fn_out, ARG);

elapsed = toc;
fprintf('Completed in %.1f minutes.\n', elapsed / 60);

end
