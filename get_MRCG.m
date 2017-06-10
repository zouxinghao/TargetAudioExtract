
function output = get_MRCG(filePath)
%This function gets the MRCG_FEATURES
addpath(['.' filesep 'MRCG_features']);
[sig, frequency] = audioread(filePath);
output_1 = MRCG_features(sig, frequency);
output = reshape(output_1.',1,numel(output_1));
