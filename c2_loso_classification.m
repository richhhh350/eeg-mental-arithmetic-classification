clear; clc; close all;
%% ===== 1. 路径与参数配置 =====
% 特征数据目录
feat_dir = fullfile(fileparts(mfilename('fullpath')), 'features');
% 结果保存目录
res_dir  = fullfile(fileparts(mfilename('fullpath')), 'results');
if ~exist(res_dir, 'dir'), mkdir(res_dir); end  % 若目录不存在则创建
% ===== 可调参数 =====
% 分段配置索引 (1~6)
%   1=1s_noOverlap, 2=1s_50pOverlap, 3=2s_noOverlap
%   4=2s_50pOverlap, 5=4s_noOverlap, 6=4s_50pOverlap
SELECTED_CONFIG = 4;  % 默认使用 2s_50pOverlap（时间分辨率与样本数均衡）
% 特征类型选择
%   1=PSD, 2=DE, 3=LEE, 4=ZCR, 5=Kurtosis, 6=Skewness, 7=Mean, 8=Variance
SELECTED_FEATURES = [2];  % 默认仅使用 DE（微分熵），76维特征向量
% MLP 默认隐藏层架构（LOSO 不单独划分验证集，使用固定架构）
MLP_LAYERS = [10];  % 单隐藏层 10 个神经元（适用于中等规模 EEG 数据）
% SVM 核函数参数（LOSO 不单独划分验证集，使用固定参数）
SVM_BOX_CONSTRAINT = 1;     % 惩罚参数 C（默认值）
SVM_KERNEL_SCALE = 'auto';  % 核尺度（MATLAB 自动启发式计算）
% 随机种子（用于 MLP 权重初始化等，保证可复现）
RNG_SEED = 42;
% 正类定义：'hi' = 高负荷(心算)为正类(Positive)，'lo' = 低负荷(静息)为负类(Negative)
POSITIVE_CLASS = 'hi';  % 将高负荷设为正类
% 条件标签映射
cond_tags = {'lo', 'hi'};  % 条件简称
cond_labels_full = {'低负荷(静息)', '高负荷(心算)'};  % 条件全称
%% ===== 2. 加载汇总特征数据 =====
fprintf('========== 留一被试法 (LOSO) 跨被试分类实验 ==========\n');
fprintf('加载汇总特征数据...\n');
summary_file = fullfile(feat_dir, 'AllFeatures_Summary.mat');
if ~exist(summary_file, 'file')
    error('汇总特征文件不存在: %s\n请先运行 b1_segmentation_feature_extraction.m', summary_file);
end
load(summary_file, 'subj_features', 'seg_configs', 'feature_type_names', ...
    'bands', 'band_names', 'CHAN_LABELS', 'n_subjects', 'subj_ids', 'n_configs', 'n_feature_types');
fprintf('已加载: %d 名被试, %d 种分段配置, %d 种特征类型\n', n_subjects, n_configs, n_feature_types);
% 显示当前参数配置
cfg = seg_configs(SELECTED_CONFIG);  % 获取选定分段配置
cfg_name = cfg.name;  % 配置名称
fprintf('\n===== 实验参数 =====\n');
fprintf('分段配置: %s (窗长 %ds, 重叠 %.0f%%)\n', cfg_name, cfg.win_len_sec, cfg.overlap_ratio * 100);
fprintf('选定特征: ');
for i = 1:length(SELECTED_FEATURES)
    fprintf('%s  ', feature_type_names{SELECTED_FEATURES(i)});
end
fprintf('\n');
fprintf('MLP 隐藏层: %s\n', mat2str(MLP_LAYERS));
fprintf('SVM 参数: C=%.2f, KernelScale=%s\n', SVM_BOX_CONSTRAINT, ...
    conditional(ischar(SVM_KERNEL_SCALE), SVM_KERNEL_SCALE, num2str(SVM_KERNEL_SCALE)));
fprintf('交叉验证方式: Leave-One-Subject-Out (%d 折)\n', n_subjects);
%% ===== 3. 初始化结果存储结构 =====
% 模型名称列表
model_names = {'LR_对数几率回归', 'MLP_多层感知机', 'SVM_支持向量机'};
n_models = length(model_names);  % 模型数量 = 3
% 评价指标名称列表
metric_names = {'ACC_准确率', 'F1_F1分数', 'Sens_灵敏度', 'Spec_特异度', 'Prec_精确率', 'NPV_阴性预测值'};
n_metrics = length(metric_names);  % 指标数量 = 6
% LOSO 各折结果存储
% loso_results{m, s} = metrics struct（第 m 个模型在第 s 折（被试）上的结果）
loso_results = cell(n_models, n_subjects);  % 模型 × 测试被试
% 混淆矩阵元素汇总（用于计算全局宏平均指标）
loso_confusion = zeros(n_models, 4, n_subjects);  % [模型, TP/TN/FP/FN, 被试]

%% ===== 4. LOSO 主循环：逐被试留出测试 =====
fprintf('\n========== 开始 LOSO 留一被试法交叉验证 ==========\n');
for test_s = 1:n_subjects  % 遍历每名被试作为测试集
    test_sub_id = subj_ids(test_s);  % 当前测试被试编号
    fprintf('\n===== LOSO 第 %d/%d 折: 测试被试 Sub%02d =====\n', ...
        test_s, n_subjects, test_sub_id);
    % 训练被试索引（除当前测试被试外的所有被试）
    train_sub_idx = setdiff(1:n_subjects, test_s);  % 训练被试索引列表
    n_train_sub = length(train_sub_idx);  % 训练被试数量
    % ---------------------------------------------------------
    % 4a. 构建训练集：汇总所有训练被试的特征数据
    % ---------------------------------------------------------
    % 首次遍历时计算特征维度
    if test_s == 1
        % 使用第一个训练被试确定特征维度
        first_train = train_sub_idx(1);
        sf_first = subj_features{SELECTED_CONFIG, first_train};  % 被试特征
        lo_feats_first = sf_first.lo_feats;  % 低负荷特征
        total_feat_dim = 0;  % 总特征维度
        chan_count = size(lo_feats_first{SELECTED_FEATURES(1)}, 2);  % 导联数 = 19
        for ff = 1:length(SELECTED_FEATURES)
            feat_idx = SELECTED_FEATURES(ff);  % 特征类型索引
            feat_sz = size(lo_feats_first{feat_idx});  % [nSeg × nCh × nSubFeat]
            total_feat_dim = total_feat_dim + chan_count * feat_sz(3);  % 累加维度
        end
        fprintf('  特征向量维度: %d (导联数=%d)\n', total_feat_dim, chan_count);
    end
    % 初始化训练集存储器（动态扩展，避免预分配不确定性）
    X_train_lo = [];  % 训练集低负荷特征
    X_train_hi = [];  % 训练集高负荷特征

    for ti = 1:n_train_sub  % 遍历训练被试
        train_s = train_sub_idx(ti);  % 当前训练被试索引
        sf_data = subj_features{SELECTED_CONFIG, train_s};  % 获取特征数据
        if isempty(sf_data)
            continue;  % 跳过空数据
        end
        lo_feats = sf_data.lo_feats;  % 低负荷特征 cell
        hi_feats = sf_data.hi_feats;  % 高负荷特征 cell
        % ----- 构建低负荷特征矩阵 -----
        n_seg_lo = size(lo_feats{SELECTED_FEATURES(1)}, 1);  % 片段数
        X_lo_tmp = zeros(n_seg_lo, total_feat_dim);  % 预分配
        col_start = 1;  % 列起始位置
        for ff = 1:length(SELECTED_FEATURES)
            feat_idx = SELECTED_FEATURES(ff);
            feat_data = lo_feats{feat_idx};  % [nSeg × nCh × nSubFeat]
            feat_2d = reshape(feat_data, n_seg_lo, []);  % 展平 [nSeg × dim]
            col_end = col_start + size(feat_2d, 2) - 1;
            X_lo_tmp(:, col_start:col_end) = feat_2d;
            col_start = col_end + 1;
        end
        X_train_lo = [X_train_lo; X_lo_tmp];  % 拼入训练集

        % ----- 构建高负荷特征矩阵 -----
        n_seg_hi = size(hi_feats{SELECTED_FEATURES(1)}, 1);  % 片段数
        X_hi_tmp = zeros(n_seg_hi, total_feat_dim);  % 预分配
        col_start = 1;
        for ff = 1:length(SELECTED_FEATURES)
            feat_idx = SELECTED_FEATURES(ff);
            feat_data = hi_feats{feat_idx};  % [nSeg × nCh × nSubFeat]
            feat_2d = reshape(feat_data, n_seg_hi, []);
            col_end = col_start + size(feat_2d, 2) - 1;
            X_hi_tmp(:, col_start:col_end) = feat_2d;
            col_start = col_end + 1;
        end
        X_train_hi = [X_train_hi; X_hi_tmp];  % 拼入训练集
    end
    % ----- 构建完整训练集 -----
    X_train = [X_train_lo; X_train_hi];  % 拼接低/高负荷训练特征
    y_train = [zeros(size(X_train_lo, 1), 1); ones(size(X_train_hi, 1), 1)];  % 标签 0/1
    fprintf('  训练集: %d 名被试, 总样本=%d (lo=%d, hi=%d)\n', ...
        n_train_sub, size(X_train, 1), size(X_train_lo, 1), size(X_train_hi, 1));
    % ---------------------------------------------------------
    % 4b. 构建测试集：仅使用当前测试被试的数据
    % ---------------------------------------------------------
    sf_test = subj_features{SELECTED_CONFIG, test_s};  % 测试被试特征
    if isempty(sf_test)
        fprintf('  警告: 被试 Sub%02d 数据为空，跳过此折\n', test_sub_id);
        continue;
    end
    lo_feats_test = sf_test.lo_feats;  % 测试被试低负荷特征
    hi_feats_test = sf_test.hi_feats;  % 测试被试高负荷特征
    % ----- 构建测试集低负荷特征 -----
    n_seg_lo_test = size(lo_feats_test{SELECTED_FEATURES(1)}, 1);  % 片段数
    X_test_lo = zeros(n_seg_lo_test, total_feat_dim);  % 预分配
    col_start = 1;
    for ff = 1:length(SELECTED_FEATURES)
        feat_idx = SELECTED_FEATURES(ff);
        feat_data = lo_feats_test{feat_idx};
        feat_2d = reshape(feat_data, n_seg_lo_test, []);
        col_end = col_start + size(feat_2d, 2) - 1;
        X_test_lo(:, col_start:col_end) = feat_2d;
        col_start = col_end + 1;
    end
    % ----- 构建测试集高负荷特征 -----
    n_seg_hi_test = size(hi_feats_test{SELECTED_FEATURES(1)}, 1);  % 片段数
    X_test_hi = zeros(n_seg_hi_test, total_feat_dim);  % 预分配
    col_start = 1;
    for ff = 1:length(SELECTED_FEATURES)
        feat_idx = SELECTED_FEATURES(ff);
        feat_data = hi_feats_test{feat_idx};
        feat_2d = reshape(feat_data, n_seg_hi_test, []);
        col_end = col_start + size(feat_2d, 2) - 1;
        X_test_hi(:, col_start:col_end) = feat_2d;
        col_start = col_end + 1;
    end
    % ----- 构建完整测试集 -----
    X_test = [X_test_lo; X_test_hi];  % 拼接低/高负荷测试特征
    y_test = [zeros(n_seg_lo_test, 1); ones(n_seg_hi_test, 1)];  % 标签 0/1
    fprintf('  测试集: 被试 Sub%02d, 样本=%d (lo=%d, hi=%d)\n', ...
        test_sub_id, size(X_test, 1), n_seg_lo_test, n_seg_hi_test);
    % ---------------------------------------------------------
    % 4c. 特征归一化（基于训练集计算 Z-score 参数，应用于测试集）
    % ---------------------------------------------------------
    [X_train_norm, norm_mu, norm_sigma] = zscore(X_train);  % 训练集标准化
    % 测试集使用训练集的均值和标准差进行归一化
    X_test_norm = (X_test - norm_mu) ./ norm_sigma;  % 测试集归一化
    X_test_norm(:, norm_sigma == 0) = 0;  % 处理常量特征（标准差为 0）
    X_train_norm(isnan(X_train_norm)) = 0;  % 替换 NaN
    % 剔除训练集中方差为零的特征
    feat_var = var(X_train_norm, 0, 1);  % 各特征方差
    valid_feat_idx = feat_var > 1e-10;  % 有效特征索引
    if ~all(valid_feat_idx)
        fprintf('  剔除了 %d 个常量特征\n', sum(~valid_feat_idx));
        X_train_norm = X_train_norm(:, valid_feat_idx);  % 保留有效特征
        X_test_norm  = X_test_norm(:, valid_feat_idx);
    end
    % ---------------------------------------------------------
    % 4d. 模型 1: 对数几率回归 (Logistic Regression)
    %     使用 fitclinear 实现 L2 正则化对数几率回归
    % ---------------------------------------------------------
    rng(RNG_SEED);  % 固定随机种子
    lr_mdl = fitclinear(X_train_norm, y_train, ...
        'Learner', 'logistic', ...     % 对数几率回归
        'Regularization', 'ridge', ... % L2 正则化
        'Lambda', 'auto', ...          % 自动选择正则化强度
        'Solver', 'lbfgs');            % L-BFGS 求解器
    % 预测并计算指标
    [y_test_pred_lr, ~] = predict(lr_mdl, X_test_norm);  % 测试集预测
    loso_results{1, test_s} = compute_all_metrics(y_test, y_test_pred_lr);  % 评估
    % ---------------------------------------------------------
    % 4e. 模型 2: MLP (多层感知机)
    %     使用固定架构（LOSO 不单独划分验证集）
    % ---------------------------------------------------------
    rng(RNG_SEED);  % 固定随机种子
    try
        mlp_mdl = fitcnet(X_train_norm, y_train, ...
            'LayerSizes', MLP_LAYERS, ...       % 隐藏层配置（固定参数）
            'Activations', 'relu', ...          % ReLU 激活函数
            'Standardize', false, ...           % 数据已手动归一化
            'Verbose', 0);                      % 关闭训练日志
        [y_test_pred_mlp, ~] = predict(mlp_mdl, X_test_norm);  % 测试集预测
        loso_results{2, test_s} = compute_all_metrics(y_test, y_test_pred_mlp);  % 评估
    catch ME  % 捕获 MLP 训练异常
        fprintf('  警告: MLP 训练失败 (%s)，使用零预测\n', ME.message);
        y_test_pred_mlp = zeros(size(y_test));  % 失败时全部预测为负类
        loso_results{2, test_s} = compute_all_metrics(y_test, y_test_pred_mlp);
    end
    % ---------------------------------------------------------
    % 4f. 模型 3: SVM (支持向量机)
    %     使用固定 RBF 核参数（LOSO 不单独划分验证集）
    % ---------------------------------------------------------
    rng(RNG_SEED);  % 固定随机种子
    try
        % 计算 KernelScale 参数
        if strcmp(SVM_KERNEL_SCALE, 'auto')
            ks_val = 'auto';  % MATLAB 自动启发式计算
        else
            ks_val = SVM_KERNEL_SCALE;  % 用户指定值
        end
        svm_mdl = fitcsvm(X_train_norm, y_train, ...
            'KernelFunction', 'rbf', ...          % RBF 核函数
            'BoxConstraint', SVM_BOX_CONSTRAINT, ... % 惩罚参数 C
            'KernelScale', ks_val, ...            % 核尺度参数
            'Standardize', false, ...             % 数据已手动归一化
            'Verbose', 0);                        % 关闭训练日志
        [y_test_pred_svm, ~] = predict(svm_mdl, X_test_norm);  % 测试集预测
        loso_results{3, test_s} = compute_all_metrics(y_test, y_test_pred_svm);  % 评估
    catch ME  % 捕获 SVM 训练异常
        fprintf('  警告: SVM 训练失败 (%s)，使用零预测\n', ME.message);
        y_test_pred_svm = zeros(size(y_test));  % 失败时全部预测为负类
        loso_results{3, test_s} = compute_all_metrics(y_test, y_test_pred_svm);
    end
    % 打印当前折三个模型的测试结果摘要
    fprintf('  ---- 本折结果 ----\n');
    fprintf('  LR:   ACC=%.4f, F1=%.4f, Sens=%.4f, Spec=%.4f, Prec=%.4f, NPV=%.4f\n', ...
        loso_results{1, test_s}.ACC, loso_results{1, test_s}.F1, ...
        loso_results{1, test_s}.Sens, loso_results{1, test_s}.Spec, ...
        loso_results{1, test_s}.Prec, loso_results{1, test_s}.NPV);
    fprintf('  MLP:  ACC=%.4f, F1=%.4f, Sens=%.4f, Spec=%.4f, Prec=%.4f, NPV=%.4f\n', ...
        loso_results{2, test_s}.ACC, loso_results{2, test_s}.F1, ...
        loso_results{2, test_s}.Sens, loso_results{2, test_s}.Spec, ...
        loso_results{2, test_s}.Prec, loso_results{2, test_s}.NPV);
    fprintf('  SVM:  ACC=%.4f, F1=%.4f, Sens=%.4f, Spec=%.4f, Prec=%.4f, NPV=%.4f\n', ...
        loso_results{3, test_s}.ACC, loso_results{3, test_s}.F1, ...
        loso_results{3, test_s}.Sens, loso_results{3, test_s}.Spec, ...
        loso_results{3, test_s}.Prec, loso_results{3, test_s}.NPV);
end
%% ===== 5. 汇总统计：计算各指标在被试群体上的均值与标准差 =====
fprintf('\n========== LOSO 汇总统计结果 ==========\n');
% 汇总统计矩阵
loso_summary_mean = zeros(n_models, n_metrics);  % 各模型×各指标均值
loso_summary_std  = zeros(n_models, n_metrics);  % 各模型×各指标标准差
for m = 1:n_models  % 遍历模型
    % 收集所有折的指标
    all_metrics_mat = [];  % [折数 × 指标数]
    valid_fold_count = 0;  % 有效折数

    for s = 1:n_subjects  % 遍历被试
        if ~isempty(loso_results{m, s})
            metrics_s = loso_results{m, s};  % 当前折的指标
            valid_fold_count = valid_fold_count + 1;  % 有效计数+1
            all_metrics_mat(valid_fold_count, :) = [  % 按顺序提取6项指标
                metrics_s.ACC, metrics_s.F1, metrics_s.Sens, ...
                metrics_s.Spec, metrics_s.Prec, metrics_s.NPV];
        end
    end
    if valid_fold_count > 0
        loso_summary_mean(m, :) = mean(all_metrics_mat, 1);  % 各指标均值
        loso_summary_std(m, :)  = std(all_metrics_mat, 0, 1);  % 各指标标准差
    end
end
% ----- 5a. 输出各折独立结果 -----
fprintf('\n========== 各折（被试）独立结果 ==========\n');
for m = 1:n_models  % 遍历模型
    fprintf('\n--- 模型: %s ---\n', model_names{m});
    fprintf('被试\tACC\tF1\tSens\tSpec\tPrec\tNPV\n');  % 表头
    for s = 1:n_subjects
        if ~isempty(loso_results{m, s})
            mt = loso_results{m, s};
            fprintf('S%02d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n', ...
                subj_ids(s), mt.ACC, mt.F1, mt.Sens, mt.Spec, mt.Prec, mt.NPV);
        else
            fprintf('S%02d\t-\t-\t-\t-\t-\t-\n', subj_ids(s));
        end
    end
    % 打印均值±标准差
    fprintf('Mean\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n', loso_summary_mean(m, :));
    fprintf('Std\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n', loso_summary_std(m, :));
end
% ----- 5b. 输出全部被试汇总表 -----
fprintf('\n========== LOSO 全部被试汇总结果（均值±标准差） ==========\n');
fprintf('%-25s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
    '模型', 'ACC', 'F1', 'Sens', 'Spec', 'Prec', 'NPV');
fprintf('%-25s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
    '', '(M±SD)', '(M±SD)', '(M±SD)', '(M±SD)', '(M±SD)', '(M±SD)');
for m = 1:n_models  % 遍历模型
    fprintf('%-25s %5.3f±%5.3f %5.3f±%5.3f %5.3f±%5.3f %5.3f±%5.3f %5.3f±%5.3f %5.3f±%5.3f\n', ...
        model_names{m}, ...
        loso_summary_mean(m, 1), loso_summary_std(m, 1), ...  % ACC
        loso_summary_mean(m, 2), loso_summary_std(m, 2), ...  % F1
        loso_summary_mean(m, 3), loso_summary_std(m, 3), ...  % Sens
        loso_summary_mean(m, 4), loso_summary_std(m, 4), ...  % Spec
        loso_summary_mean(m, 5), loso_summary_std(m, 5), ...  % Prec
        loso_summary_mean(m, 6), loso_summary_std(m, 6));     % NPV
end
%% ===== 6. 保存结果 =====
fprintf('\n========== 保存结果文件 ==========\n');
loso_result_file = fullfile(res_dir, 'LOSO_Results.mat');
save(loso_result_file, 'loso_results', 'loso_summary_mean', 'loso_summary_std', ...
    'SELECTED_CONFIG', 'SELECTED_FEATURES', 'MLP_LAYERS', 'SVM_BOX_CONSTRAINT', ...
    'SVM_KERNEL_SCALE', 'model_names', 'metric_names', 'subj_ids', 'n_subjects', '-v7.3');
fprintf('结果已保存至: %s\n', loso_result_file);
% 导出 CSV 格式结果
export_loso_csv(res_dir, 'LOSO', subj_ids, loso_results, model_names, ...
    metric_names, n_subjects, n_models);
fprintf('\n========== LOSO 留一被试法分类实验全部完成 ==========\n');
%% ===== 辅助函数 =====
function metrics = compute_all_metrics(y_true, y_pred)
    % 计算二分类全部 6 项评价指标
    % 输入:
    %   y_true - 真实标签向量 (0=低负荷, 1=高负荷)
    %   y_pred - 预测标签向量
    % 输出:
    %   metrics - 包含 ACC, F1, Sens, Spec, Prec, NPV 的结构体
    % 计算混淆矩阵的四项元素
    TP = sum(y_true == 1 & y_pred == 1);  % 真阳性: 实际高负荷且预测高负荷
    TN = sum(y_true == 0 & y_pred == 0);  % 真阴性: 实际低负荷且预测低负荷
    FP = sum(y_true == 0 & y_pred == 1);  % 假阳性: 实际低负荷但预测高负荷（第一类错误）
    FN = sum(y_true == 1 & y_pred == 0);  % 假阴性: 实际高负荷但预测低负荷（第二类错误）
    total = TP + TN + FP + FN;  % 样本总数
    % ACC: 准确率 (Accuracy) = (TP+TN) / 总样本
    if total > 0
        metrics.ACC = (TP + TN) / total;
    else
        metrics.ACC = 0;
    end
    % Sens: 灵敏度/召回率 (Sensitivity/Recall) = TP / (TP+FN)
    if (TP + FN) > 0
        metrics.Sens = TP / (TP + FN);
    else
        metrics.Sens = 0;
    end
    % Spec: 特异度 (Specificity) = TN / (TN+FP)
    if (TN + FP) > 0
        metrics.Spec = TN / (TN + FP);
    else
        metrics.Spec = 0;
    end
    % Prec: 精确率 (Precision) = TP / (TP+FP)
    if (TP + FP) > 0
        metrics.Prec = TP / (TP + FP);
    else
        metrics.Prec = 0;
    end
    % NPV: 阴性预测值 (Negative Predictive Value) = TN / (TN+FN)
    if (TN + FN) > 0
        metrics.NPV = TN / (TN + FN);
    else
        metrics.NPV = 0;
    end
    % F1: F1 分数 = 2 × Prec × Sens / (Prec + Sens)
    if (metrics.Prec + metrics.Sens) > 0
        metrics.F1 = 2 * metrics.Prec * metrics.Sens / (metrics.Prec + metrics.Sens);
    else
        metrics.F1 = 0;
    end
end
function export_loso_csv(res_dir, prefix, subj_ids, loso_results, model_names, ...
    metric_names, n_subjects, n_models)
    % 导出 LOSO 结果为 CSV 文件
    csv_rows = {};  % CSV 行数据 cell 数组
    row_idx = 0;  % 行计数器
    for m = 1:n_models  % 遍历模型
        for s = 1:n_subjects  % 遍历被试
            if ~isempty(loso_results{m, s})
                row_idx = row_idx + 1;
                mt = loso_results{m, s};
                csv_rows{row_idx, 1} = sprintf('S%02d', subj_ids(s));  % 测试被试
                csv_rows{row_idx, 2} = model_names{m};  % 模型名称
                csv_rows{row_idx, 3} = mt.ACC;   % 准确率
                csv_rows{row_idx, 4} = mt.F1;    % F1 分数
                csv_rows{row_idx, 5} = mt.Sens;  % 灵敏度
                csv_rows{row_idx, 6} = mt.Spec;  % 特异度
                csv_rows{row_idx, 7} = mt.Prec;  % 精确率
                csv_rows{row_idx, 8} = mt.NPV;   % 阴性预测值
            end
        end
    end
    % 写入 CSV
    if ~isempty(csv_rows)
        csv_table = cell2table(csv_rows, 'VariableNames', ...
            {'TestSubject', 'Model', 'ACC', 'F1', 'Sensitivity', 'Specificity', 'Precision', 'NPV'});
        csv_file = fullfile(res_dir, sprintf('%s_AllFolds.csv', prefix));
        writetable(csv_table, csv_file);
        fprintf('已导出: %s\n', csv_file);
    end
end
function s = conditional(cond, val_true, val_false)
    % 简易三元条件运算符（MATLAB 无内置三元运算符）
    if cond
        s = val_true;   % 条件为真
    else
        s = val_false;  % 条件为假
    end
end
