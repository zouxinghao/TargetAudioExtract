function [SDR, SIR, SAR] = bss_eval(signal_wav, noise_wav, sep_sig_wav, sep_noise_wav, mix_wav)
% Evaluate performance using BSS Eval 2.0
addpath(['.' filesep 'bss_eval']);
[wav_truth_signal, fs]=audioread(signal_wav);
[wav_truth_noise, fs]=audioread(noise_wav);
[wav_pred_signal, fs]=audioread(sep_sig_wav);
[wav_pred_noise, fs]=audioread(sep_noise_wav);
[wav_mix, fs]=audioread(mix_wav);

%% evaluate
sep =[wav_pred_noise , wav_pred_signal]';
orig =[wav_truth_noise , wav_truth_signal]';

[e1, e2, e3] = bss_decomp_gain(sep(2, :), 2, orig);
[SDR, SIR, SAR] = bss_crit(e1, e2, e3);

return;