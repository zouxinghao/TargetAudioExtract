
function ouotput = MRCG_features(sig, sampFreq)
% This function computes MRCG features
% sig = double(sig)
disp('I am in MRCG_features')
addpath(['.' filesep 'MRCG_features']);
sampFreq = double(sampFreq)
beta = 1000 / sqrt(sum(sig.^ 2) / length(sig));
sig = sig.* beta;
sig = reshape(sig, length(sig), 1);
g = gammatone(sig, 64, [50 8000], sampFreq); % Gammatone filterbank responses

cochlea1 = log10(cochleagram(g,sampFreq*0.020,sampFreq*0.010));
cochlea2 = log10(cochleagram(g,sampFreq*0.200,sampFreq*0.010));

M = floor(length(sig)/160);  % number of time frames
cochlea1 = cochlea1(:,1:M);
cochlea2 = cochlea2(:,1:M);

cochlea3  = get_avg(cochlea1,5,5);
cochlea4  = get_avg(cochlea1,11,11);
all_cochleas = [cochlea1; cochlea2; cochlea3; cochlea4];

% delta features is used to describe features between frames
del = deltas(all_cochleas);
ddel = deltas(deltas(all_cochleas,5),5);
ouotput = [all_cochleas;del;ddel];
