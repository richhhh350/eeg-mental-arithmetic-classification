clear; clc; close all;
%% ===== 1. 路径与参数初始化 =====
% 预处理数据路径
data_dir = fullfile(fileparts(mfilename('fullpath')), 'preprocessed_data');
% 分段数据保存路径
seg_dir  = fullfile(fileparts(mfilename('fullpath')), 'segmented_data');
% 特征数据保存路径
feat_dir = fullfile(fileparts(mfilename('fullpath')), 'features');
% 图像保存路径
fig_dir  = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(seg_dir, 'dir'), mkdir(seg_dir); end
if ~exist(feat_dir, 'dir'), mkdir(feat_dir); end
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
% 采样率与导联参数
FS = 500;                        % 采样频率 500 Hz
N_CHANNELS = 19;                 % 导联数量
CHAN_LABELS = {'Fp1','Fp2','F3','F4','F7','F8','T3','T4','C3','C4', ...
               'T5','T6','P3','P4','O1','O2','Fz','Cz','Pz'};
% 三种分片时长（秒）
seg_lengths = [1, 2, 4];         % 1秒、2秒、4秒
% 两种重叠模式
overlap_modes = [0, 0.5];        % 0=不重叠, 0.5=50%重叠
% 重叠模式标签
overlap_labels = {'noOverlap', '50pOverlap'};
% 频段定义 (Hz)
bands = struct(...
    'theta', [4 8], ...
    'alpha', [8 13], ...
    'beta',  [13 30], ...
    'gamma', [30 45]);
band_names = fieldnames(bands);   % 频段名称列表
n_bands = length(band_names);     % 频段数量
% 条件标签
cond_tags = {'lo', 'hi'};
cond_labels_full = {'低负荷(静息)', '高负荷(心算)'};
% 获取所有预处理文件列表
files = dir(fullfile(data_dir, '*_preprocessed.mat'));
% 提取被试编号（去重）
subj_ids = [];
for k = 1:length(files)
    % 文件名格式: SubXX_lo_preprocessed.mat 或 SubXX_hi_preprocessed.mat
    token = regexp(files(k).name, 'Sub(\d+)', 'tokens');
    if ~isempty(token)
        subj_ids = [subj_ids; str2double(token{1}{1})];
    end
end
subj_ids = unique(subj_ids);
n_subjects = length(subj_ids);    % 被试总数
fprintf('检测到 %d 名被试\n', n_subjects);
%% ===== 2. 分段参数预计算 =====
% 为每种分段配置计算：窗口采样点数、步长采样点数、每名被试的片段数
seg_configs = struct();           % 存储6种分段配置
config_idx = 0;                   % 配置计数器
for sl = 1:length(seg_lengths)    % 遍历3种片段长度
    for om = 1:length(overlap_modes)  % 遍历2种重叠模式
        config_idx = config_idx + 1;
        win_len_sec = seg_lengths(sl);          % 窗口时长（秒）
        win_len_samp = win_len_sec * FS;        % 窗口采样点数
        overlap_ratio = overlap_modes(om);      % 重叠比例
        step_samp = win_len_samp * (1 - overlap_ratio);  % 步长采样点数
        n_total_samp = 30000;                   % 总采样点数（60秒 × 500Hz）
        % 计算片段数量：floor((总长度-窗长)/步长) + 1
        n_segments = floor((n_total_samp - win_len_samp) / step_samp) + 1;
        % 存储配置参数
        seg_configs(config_idx).name = sprintf('%ds_%s', win_len_sec, overlap_labels{om});
        seg_configs(config_idx).win_len_sec = win_len_sec;
        seg_configs(config_idx).win_len_samp = win_len_samp;
        seg_configs(config_idx).overlap_ratio = overlap_ratio;
        seg_configs(config_idx).step_samp = step_samp;
        seg_configs(config_idx).n_segments = n_segments;
        fprintf('配置 %d [%s]: 窗长=%ds(%d样点), 重叠=%.0f%%, 步长=%d样点, 片段数=%d\n', ...
            config_idx, seg_configs(config_idx).name, win_len_sec, win_len_samp, ...
            overlap_ratio*100, step_samp, n_segments);
    end
end
n_configs = config_idx;           % 分段配置总数（=6）
%% ===== 3. 初始化特征存储结构 =====
feature_type_names = {
    'PSD_频段平均功率', ...
    'DE_微分熵', ...
    'LogEnergyEntropy_对数能量熵', ...
    'ZCR_过零率', ...
    'Kurtosis_峰度', ...
    'Skewness_偏度', ...
    'Mean_均值', ...
    'Variance_方差'};
n_feature_types = length(feature_type_names);
% 为每种分段配置预分配特征存储
% pooled_features{c}：汇总所有被试的特征
% subj_features{c}{s}：第s名被试的特征
pooled_features = cell(n_configs, 1);
subj_features = cell(n_configs, n_subjects);
for c = 1:n_configs
    pooled_features{c} = struct();
    pooled_features{c}.lo = cell(n_feature_types, 1);
    pooled_features{c}.hi = cell(n_feature_types, 1);
    for f = 1:n_feature_types
        pooled_features{c}.lo{f} = [];  % 低负荷条件汇总
        pooled_features{c}.hi{f} = [];  % 高负荷条件汇总
    end
end
%% ===== 4. 主处理循环：逐被试、逐条件进行分段与特征提取 =====
% 创建总体进度提示
fprintf('\n========== 开始分段与特征提取 ==========\n');
% 实际处理时，每个被试有hi和lo两个条件文件
for s = 1:n_subjects
    sub_id = subj_ids(s);
    fprintf('\n--- 处理被试 Sub%02d (%d/%d) ---\n', sub_id, s, n_subjects);
    for cond = 1:2  % 遍历lo和hi两种条件
        cond_tag = cond_tags{cond};
        % 构造预处理文件名并加载
        mat_file = fullfile(data_dir, sprintf('Sub%02d_%s_preprocessed.mat', sub_id, cond_tag));
        if ~exist(mat_file, 'file')
            fprintf('  警告: 文件 %s 不存在，跳过\n', mat_file);
            continue;
        end
        load(mat_file, 'EEG', 'preprocessing_info');
        % ---- 4a. 将Epoch数据拼接回连续信号 ----
        % EEG.data维度: [导联数 × 采样点/epoch × epoch数] = [19 × 1000 × 30]
        % 沿第2维方向在第3维拼接，得到连续信号 [19 × 30000]
        eeg_continuous = reshape(EEG.data, N_CHANNELS, []);
        n_total_samp = size(eeg_continuous, 2);  % 应为30000
        % ---- 4b. 对每种分段配置进行分片段 ----
        for c = 1:n_configs
            cfg = seg_configs(c);
            win_samp = cfg.win_len_samp;    % 窗口采样点数
            step_samp = cfg.step_samp;      % 步长采样点数
            n_seg = cfg.n_segments;         % 片段数量
            % 初始化当前配置下的分段数据矩阵
            % segmented_data: [导联数 × 窗口采样点 × 片段数]
            segmented_data = zeros(N_CHANNELS, win_samp, n_seg);
            for seg = 1:n_seg
                % 计算当前片段的起始和结束采样索引
                start_idx = (seg - 1) * step_samp + 1;
                end_idx = start_idx + win_samp - 1;
                % 截取所有导联在该时间窗口内的数据（同步截取）
                segmented_data(:, :, seg) = eeg_continuous(:, start_idx:end_idx);
            end
            % ---- 4c. 保存分段数据集 ----
            seg_file = fullfile(seg_dir, sprintf('Sub%02d_%s_%s.mat', ...
                sub_id, cond_tag, cfg.name));
            seg_params = cfg;  % 分段参数
            save(seg_file, 'segmented_data', 'seg_params', 'CHAN_LABELS', '-v7.3');
            % ---- 4d. 提取8种特征 ----
            % 对每个片段、每个导联分别提取特征
            % 预分配特征矩阵：[片段数 × 导联数 × 各特征子维度]
            feat_psd  = zeros(n_seg, N_CHANNELS, n_bands);  % PSD特征
            feat_de   = zeros(n_seg, N_CHANNELS, n_bands);  % 微分熵特征
            feat_lee  = zeros(n_seg, N_CHANNELS, n_bands);  % 对数能量熵特征
            feat_zcr  = zeros(n_seg, N_CHANNELS, 1);        % 过零率特征
            feat_kurt = zeros(n_seg, N_CHANNELS, 1);        % 峰度特征
            feat_skew = zeros(n_seg, N_CHANNELS, 1);        % 偏度特征
            feat_mean = zeros(n_seg, N_CHANNELS, 1);        % 均值特征
            feat_var  = zeros(n_seg, N_CHANNELS, 1);        % 方差特征
            % 预计算频段索引
            band_idx_cache = cell(n_bands, 1);
            % 使用FFT周期图法计算PSD，FFT点数取大于等于信号长度的最小2的幂次
            nfft_psd = 2^nextpow2(win_samp);  % FFT点数（2的幂次，加速计算）
            for seg = 1:n_seg
                for ch = 1:N_CHANNELS
                    % 提取当前片段、当前导联的信号
                    x = squeeze(segmented_data(ch, :, seg))';
                    % ============ 计算PSD（傅里叶变换周期图法）============
                    % 对信号做N点FFT，得到双边频谱
                    X_fft = fft(x, nfft_psd);
                    % 取单边频谱（仅保留正频率部分，含直流分量）
                    X_single = X_fft(1 : nfft_psd/2 + 1);
                    % 周期图法功率谱密度：PSD = |X(f)|² / (N * fs)
                    % N为原始信号长度，fs为采样率，结果单位为 μV²/Hz
                    pxx = (abs(X_single).^2) / (win_samp * FS);
                    % 直流和Nyquist频率分量需要特殊处理（功率加倍修正）
                    pxx(2 : end-1) = 2 * pxx(2 : end-1);
                    % 生成对应的频率轴（单位Hz）
                    f = (0 : nfft_psd/2)' * (FS / nfft_psd);
                    df = f(2) - f(1);  % 频率分辨率 = fs/NFFT
                    % 对每个频段：基于PSD推导所有频域特征
                    for b = 1:n_bands
                        band_range = bands.(band_names{b});
                        % 频段索引
                        if isempty(band_idx_cache{b})
                            band_idx_cache{b} = (f >= band_range(1)) & (f <= band_range(2));
                        end
                        bidx = band_idx_cache{b};
                        pxx_band = pxx(bidx);  % 频段内PSD值
                        % ---- 特征1: PSD频段平均功率 ----
                        feat_psd(seg, ch, b) = mean(pxx_band);
                        % ---- 特征2: 微分熵 DE = 0.5*log2(2*pi*e*var_band) ----
                        % 由Parseval定理: var_band = sum(PSD_band)*df
                        var_band = sum(pxx_band) * df;
                        if var_band > 1e-15
                            feat_de(seg, ch, b) = 0.5 * log2(2 * pi * exp(1) * var_band);
                        else
                            feat_de(seg, ch, b) = -50;  % 接近0时的下界
                        end
                        % ---- 特征3: 对数能量熵 LEE = log10(energy_band) ----
                        % energy_band = sum(x_band.^2) = N * var_band
                        energy_band = win_samp * var_band;
                        if energy_band > 1e-15
                            feat_lee(seg, ch, b) = log10(energy_band);
                        else
                            feat_lee(seg, ch, b) = -15;  % 接近0时的下界
                        end
                    end
                    % ---- 特征4: 过零率----
                    % 统计信号穿越零点的次数，反映信号频率特性
                    zc_count = sum(abs(diff(sign(x - mean(x))))) / 2;
                    feat_zcr(seg, ch) = zc_count / length(x);  % 归一化过零率
                    % ---- 特征5: 峰度 (Kurtosis) ----
                    feat_kurt(seg, ch) = kurtosis(x);
                    % ---- 特征6: 偏度 (Skewness) ----
                    feat_skew(seg, ch) = skewness(x);
                    % ---- 特征7: 均值 (Mean) ----
                    feat_mean(seg, ch) = mean(x);
                    % ---- 特征8: 方差 (Variance) ----
                    feat_var(seg, ch) = var(x);
                end
            end
            % ---- 4e. 将特征累加到汇总存储结构中 ----
            % pooled_features用于全体箱型图（汇总所有被试）
            % subj_features用于被试依赖箱型图（每个被试单独）
            % 将所有特征存入一个cell数组方便遍历
            feats_all = {feat_psd, feat_de, feat_lee, feat_zcr, feat_kurt, feat_skew, feat_mean, feat_var};
            for f = 1:n_feature_types
                % 将当前特征数据重排为 [片段数*导联数 × 子特征数]
                feat_data = feats_all{f};
                feat_2d = reshape(feat_data, n_seg * N_CHANNELS, []);
                % 累加到汇总池（区分lo/hi条件）
                if strcmp(cond_tag, 'lo')
                    pooled_features{c}.lo{f} = [pooled_features{c}.lo{f}; feat_2d];
                else
                    pooled_features{c}.hi{f} = [pooled_features{c}.hi{f}; feat_2d];
                end
            end
            % ---- 4f. 保存单个被试的特征数据 ----
            feat_file = fullfile(feat_dir, sprintf('Sub%02d_%s_%s_features.mat', ...
                sub_id, cond_tag, cfg.name));
            save(feat_file, 'feat_psd', 'feat_de', 'feat_lee', 'feat_zcr', ...
                'feat_kurt', 'feat_skew', 'feat_mean', 'feat_var', ...
                'seg_params', 'bands', 'CHAN_LABELS', 'feature_type_names', '-v7.3');
            % 存储被试级别特征（用于被试依赖箱型图）
            if strcmp(cond_tag, 'lo')
                if isempty(subj_features{c, s})
                    subj_features{c, s} = struct();
                end
                subj_features{c, s}.lo_feats = feats_all;
                subj_features{c, s}.n_seg = n_seg;
            else
                if isempty(subj_features{c, s})
                    subj_features{c, s} = struct();
                end
                subj_features{c, s}.hi_feats = feats_all;
                subj_features{c, s}.n_seg = n_seg;
            end
        end
        fprintf('  条件 %s 处理完成\n', cond_tag);
    end
end
%% ===== 5. 保存汇总特征集（48个特征集） =====
fprintf('\n========== 保存汇总特征集 ==========\n');
% 对每种分段配置、每种特征类型，生成一个独立的特征集文件
% 特征集命名规则: FeatureSet_<分段配置>_<特征类型>.mat
for c = 1:n_configs
    cfg_name = seg_configs(c).name;
    for f = 1:n_feature_types
        % 提取当前配置、当前特征类型的lo和hi汇总数据
        feat_lo = pooled_features{c}.lo{f};
        feat_hi = pooled_features{c}.hi{f};
        % 构造特征集元信息
        feature_set_info.config_name = cfg_name;
        feature_set_info.win_len_sec = seg_configs(c).win_len_sec;
        feature_set_info.overlap_ratio = seg_configs(c).overlap_ratio;
        feature_set_info.feature_type = feature_type_names{f};
        feature_set_info.feature_index = f;
        feature_set_info.bands = bands;
        feature_set_info.n_subjects = n_subjects;
        feature_set_info.n_channels = N_CHANNELS;
        feature_set_info.channel_labels = CHAN_LABELS;
        % 保存特征集
        feat_set_file = fullfile(feat_dir, sprintf('FeatureSet_%s_%s.mat', cfg_name, feature_type_names{f}));
        save(feat_set_file, 'feat_lo', 'feat_hi', 'feature_set_info', '-v7.3');
        % 打印特征集信息
        fprintf('  特征集: %s | 低负荷样本数=%d | 高负荷样本数=%d | 子特征维度=%d\n', ...
            feature_set_info.feature_type, size(feat_lo,1), size(feat_hi,1), size(feat_lo,2));
    end
end
%% ===== 6. 保存全部汇总数据 =====
fprintf('\n========== 保存汇总数据 ==========\n');
summary_file = fullfile(feat_dir, 'AllFeatures_Summary.mat');
save(summary_file, 'pooled_features', 'subj_features', 'seg_configs', ...
    'feature_type_names', 'bands', 'band_names', 'CHAN_LABELS', ...
    'n_subjects', 'n_configs', 'n_feature_types', 'cond_tags', 'subj_ids', '-v7.3');
fprintf('汇总数据已保存至: %s\n', summary_file);

fprintf('\n========== 分段与特征提取全部完成 ==========\n');
