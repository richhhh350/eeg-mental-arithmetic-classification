function validate_matlab_de(input_path, output_path)
% Same-input reference for b1_segmentation_feature_extraction.m.
% input: windows [window x channel x sample], fs, bands [band x 2].
% This reproduces V1's raw (not demeaned) periodogram and DE formula.
% It intentionally does not load or execute the legacy preprocessing code.
loaded = load(input_path, 'windows', 'fs', 'bands');
windows = double(loaded.windows);
fs = double(loaded.fs);
bands = double(loaded.bands);
n_windows = size(windows, 1);
n_channels = size(windows, 2);
n_samples = size(windows, 3);
n_bands = size(bands, 1);
nfft = 2 ^ nextpow2(n_samples);
f = (0:nfft / 2)' * (fs / nfft);
df = fs / nfft;
de_matlab = zeros(n_windows, n_channels, n_bands);
power_matlab = zeros(n_windows, n_channels, n_bands);
for w = 1:n_windows
    for ch = 1:n_channels
        x = squeeze(windows(w, ch, :));
        X_fft = fft(x, nfft);
        pxx = abs(X_fft(1:nfft / 2 + 1)) .^ 2 / (n_samples * fs);
        pxx(2:end - 1) = 2 * pxx(2:end - 1);
        for b = 1:n_bands
            mask = f >= bands(b, 1) & f <= bands(b, 2);
            power = sum(pxx(mask)) * df;
            power_matlab(w, ch, b) = power;
            if power > 1e-15
                de_matlab(w, ch, b) = 0.5 * log2(2 * pi * exp(1) * power);
            else
                de_matlab(w, ch, b) = -50;
            end
        end
    end
end
matlab_version = version;
save(output_path, 'de_matlab', 'power_matlab', 'matlab_version', '-v7');
fprintf('Validated input shape: %d x %d x %d; %d bands\n', ...
    n_windows, n_channels, n_samples, n_bands);
end
