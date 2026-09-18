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
%   使用方括号数组指定多个特征类型（例：[1,2,3] = PSD+DE+LEE合并）
SELECTED_FEATURES = [2];  % 默认仅使用 DE（微分熵），76维特征向量
% 数据集划分比例 [训练 验证 测试]（须满足三者之和 = 1.0）
SPLIT_RATIO = [0.60, 0.20, 0.20];  % 训练:验证:测试 = 60%:20%:20%
% 是否使用 PCA 降维（当特征维度 > 样本数两倍时建议开启）
USE_PCA = false;  % 不使用 PCA（DE 特征 76 维，样本数 59~119，维度合理）
PCA_VARIANCE = 0.95;  % PCA 保留方差比例（仅 USE_PCA=true 时生效）
% 随机种子（用于 MLP 初始化、数据打乱等，保证可复现）
RNG_SEED = 42;
% 正类定义：'hi' = 高负荷(心算)为正类(Positive)，'lo' = 低负荷(静息)为负类(Negative)
POSITIVE_CLASS = 'hi';  % 将高负荷设为正类
NEGATIVE_CLASS = 'lo';  % 将低负荷设为负类
% 条件标签映射
cond_tags = {'lo', 'hi'}; 
cond_labels_full = {'低负荷(静息)', '高负荷(心算)'};  % 条件全称
%% ===== 2. 加载汇总特征数据 =====
fprintf('========== 被试依赖分类实验 ==========\n');
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
fprintf('划分比例: 训练 %.0f%% / 验证 %.0f%% / 测试 %.0f%%\n', SPLIT_RATIO(1)*100, SPLIT_RATIO(2)*100, SPLIT_RATIO(3)*100);
fprintf('PCA 降维: %s\n', conditional(USE_PCA, '启用', '禁用'));
fprintf('正类标签: %s (%s)\n', POSITIVE_CLASS, cond_labels_full{2});
%% ===== 3. 初始化结果存储结构 =====
% 模型名称列表
model_names = {'LR_对数几率回归', 'MLP_多层感知机', 'SVM_支持向量机'};
n_models = length(model_names);  % 模型数量 = 3
% 评价指标名称列表
metric_names = {'ACC_准确率', 'F1_F1分数', 'Sens_灵敏度', 'Spec_特异度', 'Prec_精确率', 'NPV_阴性预测值'};
n_metrics = length(metric_names);  % 指标数量 = 6
% 数据集划分名称
split_names = {'train_训练集', 'val_验证集', 'test_测试集'};
n_splits = length(split_names);  % 划分数量 = 3
% 各被试在 训练/验证/测试 集上的结果
% subject_results{s, m, sp} = struct with metric fields
%   维度: 被试 × 模型 × 数据集划分
subject_results = cell(n_subjects, n_models, n_splits);
% MLP 最优架构记录（每名被试的最简架构）
mlp_architectures = cell(n_subjects, 1);  % 记录每名被试选中的 MLP 隐藏层结构
% SVM 最优参数记录
svm_params = cell(n_subjects, 1);  % 记录每名被试的 (BoxConstraint, KernelScale)
%% ===== 4. 逐被试分类实验 =====
fprintf('\n========== 开始逐被试分类实验 ==========\n');
for s = 1:n_subjects  % 遍历所有被试
    sub_id = subj_ids(s);  % 当前被试编号
    fprintf('\n===== 被试 Sub%02d (%d/%d) =====\n', sub_id, s, n_subjects);
    % ---------------------------------------------------------
    % 4a. 提取当前被试的特征数据并构建特征矩阵
    % ---------------------------------------------------------
    % 从 subj_features 提取指定配置和特征类型的数据
    sf_data = subj_features{SELECTED_CONFIG, s};  % 当前被试在当前配置下的特征
    if isempty(sf_data)
        fprintf('  警告: 被试 Sub%02d 数据为空，跳过\n', sub_id);
        continue;  % 跳过无数据的被试
    end
    % 获取低负荷(lo)和高负荷(hi)条件的特征数据
    lo_feats = sf_data.lo_feats;  % 低负荷特征 cell 数组，每个元素 [nSeg × nCh × nSubFeat]
    hi_feats = sf_data.hi_feats;  % 高负荷特征 cell 数组
    % 首次遍历时构建特征维度信息
    if s == 1
        total_feat_dim = 0;  % 总特征维度计数器
        chan_count = size(lo_feats{SELECTED_FEATURES(1)}, 2);  % 导联数量
        for ff = 1:length(SELECTED_FEATURES)
            feat_idx = SELECTED_FEATURES(ff);  % 特征类型索引
            feat_sz = size(lo_feats{feat_idx});  % [nSeg × nCh × nSubFeat]
            total_feat_dim = total_feat_dim + chan_count * feat_sz(3);  % 累加维度
        end
        fprintf('  特征向量维度: %d (导联数=%d)\n', total_feat_dim, chan_count);
    end
    % ----- 构建低负荷特征矩阵 X_lo -----
    n_seg_lo = size(lo_feats{SELECTED_FEATURES(1)}, 1);  % 低负荷片段数量
    X_lo = zeros(n_seg_lo, total_feat_dim);  % 预分配低负荷特征矩阵
    col_start = 1;  % 列起始位置
    for ff = 1:length(SELECTED_FEATURES)  % 遍历选定的特征类型
        feat_idx = SELECTED_FEATURES(ff);  % 特征类型索引
        feat_data = lo_feats{feat_idx};  % [nSeg × nCh × nSubFeat]
        feat_2d = reshape(feat_data, n_seg_lo, []);  % 展平为 [nSeg × (nCh*nSubFeat)]
        col_end = col_start + size(feat_2d, 2) - 1;  % 列结束位置
        X_lo(:, col_start:col_end) = feat_2d;  % 填入特征矩阵
        col_start = col_end + 1;  % 更新列起始位置
    end
    % ----- 构建高负荷特征矩阵 X_hi -----
    n_seg_hi = size(hi_feats{SELECTED_FEATURES(1)}, 1);  % 高负荷片段数量
    X_hi = zeros(n_seg_hi, total_feat_dim);  % 预分配高负荷特征矩阵
    col_start = 1;  % 列起始位置
    for ff = 1:length(SELECTED_FEATURES)
        feat_idx = SELECTED_FEATURES(ff);
        feat_data = hi_feats{feat_idx};  % [nSeg × nCh × nSubFeat]
        feat_2d = reshape(feat_data, n_seg_hi, []);  % 展平
        col_end = col_start + size(feat_2d, 2) - 1;
        X_hi(:, col_start:col_end) = feat_2d;
        col_start = col_end + 1;
    end
    % ----- 构建完整数据集 X 和标签 y -----
    % 标签: 0 = 低负荷(lo), 1 = 高负荷(hi)
    X = [X_lo; X_hi];  % 拼接低负荷和高负荷特征，先 lo 后 hi
    y = [zeros(n_seg_lo, 1); ones(n_seg_hi, 1)];  % 对应标签
    % 创建时间索引向量（每个条件下片段按采集时间递增排列）
    time_idx_lo = (1:n_seg_lo)';  % lo 条件下片段的时间序号
    time_idx_hi = (1:n_seg_hi)';  % hi 条件下片段的时间序号
    time_idx = [time_idx_lo; time_idx_hi];  % 拼接时间索引
    fprintf('  样本数: lo=%d, hi=%d, 总计=%d\n', n_seg_lo, n_seg_hi, size(X,1));
    % ---------------------------------------------------------
    % 4b. 数据集划分：留出法 + 分层抽样 + 时序规则
    %     对每个类别独立按时间切分，保证训练集时间早于测试集
    % ---------------------------------------------------------
    % 计算各划分在每类中的边界索引
    train_ratio = SPLIT_RATIO(1);  % 训练集比例
    val_ratio   = SPLIT_RATIO(2);  % 验证集比例
    % test_ratio = SPLIT_RATIO(3);  % 测试集比例（= 1 - train - val）
    % lo 条件的划分边界（按时间先后划分）
    lo_train_end = round(n_seg_lo * train_ratio);  % lo 训练集结束索引
    lo_val_end   = round(n_seg_lo * (train_ratio + val_ratio));  % lo 验证集结束索引
    % hi 条件的划分边界
    hi_train_end = round(n_seg_hi * train_ratio);  % hi 训练集结束索引
    hi_val_end   = round(n_seg_hi * (train_ratio + val_ratio));  % hi 验证集结束索引
    % 为每个样本分配数据集标签 (1=train, 2=val, 3=test)
    split_label = zeros(size(y));  % 初始化划分标签向量
    for i = 1:size(X, 1)  % 遍历每个样本
        if y(i) == 0  % 低负荷样本
            if time_idx(i) <= lo_train_end
                split_label(i) = 1;  % 训练集
            elseif time_idx(i) <= lo_val_end
                split_label(i) = 2;  % 验证集
            else
                split_label(i) = 3;  % 测试集
            end
        else  % 高负荷样本
            if time_idx(i) <= hi_train_end
                split_label(i) = 1;  % 训练集
            elseif time_idx(i) <= hi_val_end
                split_label(i) = 2;  % 验证集
            else
                split_label(i) = 3;  % 测试集
            end
        end
    end
    % 提取各划分的数据
    X_train = X(split_label == 1, :);  % 训练集特征
    y_train = y(split_label == 1);     % 训练集标签
    X_val   = X(split_label == 2, :);  % 验证集特征
    y_val   = y(split_label == 2);     % 验证集标签
    X_test  = X(split_label == 3, :);  % 测试集特征
    y_test  = y(split_label == 3);     % 测试集标签
    fprintf('  划分: 训练=%d, 验证=%d, 测试=%d\n', ...
        length(y_train), length(y_val), length(y_test));
    % 检查划分有效性
    if isempty(X_train) || isempty(X_val) || isempty(X_test)
        fprintf('  错误: 数据划分后存在空集，跳過该被试\n');
        continue;
    end
    % 检查各类别在各划分中的分布
    fprintf('  训练集 lo/hi: %d/%d | 验证集 lo/hi: %d/%d | 测试集 lo/hi: %d/%d\n', ...
        sum(y_train == 0), sum(y_train == 1), ...
        sum(y_val == 0), sum(y_val == 1), ...
        sum(y_test == 0), sum(y_test == 1));
    % ---------------------------------------------------------
    % 4c. 特征归一化（Z-score，基于训练集计算参数）
    % ---------------------------------------------------------
    [X_train_norm, norm_mu, norm_sigma] = zscore(X_train);  % 训练集标准化并记录均值/标准差
    % 对验证集和测试集使用训练集的归一化参数
    X_val_norm  = (X_val - norm_mu) ./ norm_sigma;   % 验证集归一化
    X_test_norm = (X_test - norm_mu) ./ norm_sigma;  % 测试集归一化
    % 处理标准差为 0 的特征（常量特征）避免除零
    X_val_norm(:, norm_sigma == 0) = 0;   % 替换 NaN 为 0
    X_test_norm(:, norm_sigma == 0) = 0;
    X_train_norm(isnan(X_train_norm)) = 0;  % 替换训练集中可能的 NaN
    % 可选 PCA 降维（在已归一化的数据上执行）
    if USE_PCA
        [coeff, X_train_pca, ~, ~, explained] = pca(X_train_norm);  % 对训练集执行 PCA
        % 确定保留的主成分数量
        cum_var = cumsum(explained);  % 累计方差解释比例
        n_components = find(cum_var >= PCA_VARIANCE * 100, 1);  % 达到设定方差阈值的最少成分数
        fprintf('  PCA: %d 维 -> %d 维 (保留 %.1f%% 方差)\n', size(X_train_norm,2), n_components, cum_var(n_components));
        % 投影到主成分空间（训练/验证/测试集均已归一化，直接投影）
        X_train_norm = X_train_pca(:, 1:n_components);  % 训练集降维
        X_val_norm   = X_val_norm * coeff(:, 1:n_components);   % 验证集降维
        X_test_norm  = X_test_norm * coeff(:, 1:n_components);  % 测试集降维
    end
    % 剔除常量特征（方差为零的特征对分类无贡献且可能导致数值问题）
    feat_var = var(X_train_norm, 0, 1);  % 计算每列特征的方差
    valid_feat_idx = feat_var > 1e-10;  % 方差大于极小值的特征
    if ~all(valid_feat_idx)
        fprintf('  剔除了 %d 个常量特征\n', sum(~valid_feat_idx));
        X_train_norm = X_train_norm(:, valid_feat_idx);  % 保留有效特征
        X_val_norm   = X_val_norm(:, valid_feat_idx);
        X_test_norm  = X_test_norm(:, valid_feat_idx);
    end
    % ---------------------------------------------------------
    % 4d. 模型 1: 对数几率回归 (Logistic Regression)
    %     使用 fitclinear 实现 L2 正则化的对数几率回归
    % ---------------------------------------------------------
    rng(RNG_SEED);  % 固定随机种子
    % 训练对数几率回归模型（线性分类器 + L2 正则化）
    lr_mdl = fitclinear(X_train_norm, y_train, ...
        'Learner', 'logistic', ...     % 使用对数几率回归（逻辑回归）
        'Regularization', 'ridge', ... % L2 正则化（Ridge）
        'Lambda', 'auto', ...          % 自动选择正则化强度
        'Solver', 'lbfgs');            % L-BFGS 求解器（适用于中小规模数据）
    % 在训练集、验证集、测试集上预测
    [y_train_pred_lr, ~] = predict(lr_mdl, X_train_norm);  % 训练集预测
    [y_val_pred_lr,   ~] = predict(lr_mdl, X_val_norm);    % 验证集预测
    [y_test_pred_lr,  ~] = predict(lr_mdl, X_test_norm);   % 测试集预测
    % 计算并存储各项评价指标
    lr_metrics_train = compute_all_metrics(y_train, y_train_pred_lr);  % 训练集指标
    lr_metrics_val   = compute_all_metrics(y_val,   y_val_pred_lr);    % 验证集指标
    lr_metrics_test  = compute_all_metrics(y_test,  y_test_pred_lr);   % 测试集指标
    % 存入结果结构体
    subject_results{s, 1, 1} = lr_metrics_train;  % LR 训练集结果
    subject_results{s, 1, 2} = lr_metrics_val;    % LR 验证集结果
    subject_results{s, 1, 3} = lr_metrics_test;   % LR 测试集结果
    % ---------------------------------------------------------
    % 4e. 模型 2: MLP (多层感知机) — 验证集搜索最简网络架构
    %     使用 fitcnet 实现，在验证集上评估不同架构，
    %     选择验证 ACC 最高、同时参数量最少的架构
    % ---------------------------------------------------------
    % 定义待搜索的 MLP 隐藏层架构候选列表
    % 每个 cell 元素为一个向量，表示各隐藏层的神经元数量
    mlp_arch_candidates = {
        [],            ...  % 无隐藏层（等价于线性模型）
        [5],           ...  % 1 隐藏层 5 个神经元
        [10],          ...  % 1 隐藏层 10 个神经元
        [20],          ...  % 1 隐藏层 20 个神经元
        [50],          ...  % 1 隐藏层 50 个神经元
        [5, 5],        ...  % 2 隐藏层各 5 个神经元
        [10, 5],       ...  % 2 隐藏层 10+5 个神经元
        [10, 10],      ...  % 2 隐藏层各 10 个神经元
        [20, 10]       ...  % 2 隐藏层 20+10 个神经元
    };
    n_arch = length(mlp_arch_candidates);  % 候选架构总数
    % 预分配架构评估结果存储
    arch_val_acc = zeros(n_arch, 1);    % 每种架构的验证集准确率
    arch_n_params = zeros(n_arch, 1);   % 每种架构的总参数量
    arch_models = cell(n_arch, 1);      % 每种架构训练出的模型
    fprintf('  MLP 架构搜索 (%d 个候选):\n', n_arch);
    for a = 1:n_arch  % 遍历所有候选架构
        arch_layers = mlp_arch_candidates{a};  % 当前架构的隐藏层配置
        if isempty(arch_layers)
            arch_name = '线性(无隐藏层)';  % 显示名称
        else
            arch_name = mat2str(arch_layers);  % 转换为字符串显示
        end
        rng(RNG_SEED);  % 固定随机种子保证每个架构可复现
        try
            % 训练 MLP 分类器
            mlp_tmp = fitcnet(X_train_norm, y_train, ...
                'LayerSizes', arch_layers, ...          % 隐藏层配置
                'Activations', 'relu', ...              % ReLU 激活函数
                'Standardize', false, ...               % 数据已手动归一化
                'Verbose', 0, ...                       % 关闭训练日志
                'ValidationData', {X_val_norm, y_val}); % 使用验证集监控早停
            % 在验证集上评估准确率
            [y_pred_tmp, ~] = predict(mlp_tmp, X_val_norm);
            arch_val_acc(a) = mean(y_pred_tmp == y_val);  % 计算验证准确率
            % 计算总参数量（weights + biases）
            n_feat = size(X_train_norm, 2);  % 输入特征维度
            n_params_total = 0;  % 参数计数
            prev_size = n_feat;  % 上一层尺寸（起始为输入层）
            for layer_idx = 1:length(arch_layers)  % 遍历每层
                cur_size = arch_layers(layer_idx);         % 当前层神经元数
                n_params_total = n_params_total + prev_size * cur_size + cur_size;  % 权重+偏置
                prev_size = cur_size;                      % 更新上一层尺寸
            end
            n_params_total = n_params_total + prev_size * 2 + 2;  % 输出层（二分类）
            arch_n_params(a) = n_params_total;  % 记录参数量

            arch_models{a} = mlp_tmp;  % 存储训练好的模型
            fprintf('    [%d/%d] %s: Val ACC=%.4f, 参数=%d\n', ...
                a, n_arch, arch_name, arch_val_acc(a), arch_n_params(a));
        catch ME  % 捕获训练异常
            fprintf('    [%d/%d] %s: 训练失败 (%s)\n', a, n_arch, arch_name, ME.message);
            arch_val_acc(a) = 0;  % 标记为失败
            arch_n_params(a) = inf;  % 参数量设极大值
        end
    end
    % 选择最优架构：先找最高验证准确率，再在准确率平局时选参数量最少
    [best_acc, ~] = max(arch_val_acc);  % 最高验证准确率
    acc_tolerance = 0.005;  % 准确率容差（0.5%以内视为平局）
    near_best_idx = find(arch_val_acc >= best_acc - acc_tolerance);  % 接近最优的候选
    [~, min_params_idx] = min(arch_n_params(near_best_idx));  % 其中参数量最少的
    best_arch_idx = near_best_idx(min_params_idx);  % 最终选定的架构索引

    best_mlp_mdl = arch_models{best_arch_idx};  % 最优 MLP 模型
    best_arch_layers = mlp_arch_candidates{best_arch_idx};  % 最优架构的隐藏层配置
    % 记录该被试的最优 MLP 架构
    mlp_architectures{s} = best_arch_layers;
    if isempty(best_arch_layers)
        fprintf('  -> 最优 MLP: 线性(无隐藏层), Val ACC=%.4f, 参数=%d\n', ...
            arch_val_acc(best_arch_idx), arch_n_params(best_arch_idx));
    else
        fprintf('  -> 最优 MLP: %s, Val ACC=%.4f, 参数=%d\n', ...
            mat2str(best_arch_layers), arch_val_acc(best_arch_idx), arch_n_params(best_arch_idx));
    end
    % 在全部三个数据集上评估最优 MLP
    [y_train_pred_mlp, ~] = predict(best_mlp_mdl, X_train_norm);  % 训练集
    [y_val_pred_mlp,   ~] = predict(best_mlp_mdl, X_val_norm);    % 验证集
    [y_test_pred_mlp,  ~] = predict(best_mlp_mdl, X_test_norm);   % 测试集
    mlp_metrics_train = compute_all_metrics(y_train, y_train_pred_mlp);
    mlp_metrics_val   = compute_all_metrics(y_val,   y_val_pred_mlp);
    mlp_metrics_test  = compute_all_metrics(y_test,  y_test_pred_mlp);
    subject_results{s, 2, 1} = mlp_metrics_train;  % MLP 训练集
    subject_results{s, 2, 2} = mlp_metrics_val;    % MLP 验证集
    subject_results{s, 2, 3} = mlp_metrics_test;   % MLP 测试集
    % ---------------------------------------------------------
    % 4f. 模型 3: SVM (支持向量机) — 验证集优化超参数
    %     使用 fitcsvm 实现 RBF 核 SVM，网格搜索 C 和 γ
    % ---------------------------------------------------------
    % 定义超参数搜索网格
    svm_C_grid = [0.01, 0.1, 1, 10, 100];  % 惩罚参数 C（BoxConstraint）
    svm_gamma_grid = [0.01, 0.1, 1, 10, 100];  % RBF 核参数 γ（KernelScale 的倒数关系）
    n_C = length(svm_C_grid);      % C 参数候选数
    n_gamma = length(svm_gamma_grid);  % γ 参数候选数
    % 预分配网格搜索结果矩阵
    svm_val_acc_grid = zeros(n_C, n_gamma);  % 网格搜索的验证准确率
    fprintf('  SVM 超参数网格搜索 (%d×%d=%d 组):\n', n_C, n_gamma, n_C * n_gamma);
    best_svm_acc = 0;  % 记录最佳验证准确率
    best_svm_C = 1;    % 最佳 C 参数
    best_svm_ks = 1;   % 最佳 KernelScale 参数
    best_svm_mdl = []; % 最佳 SVM 模型
    for ci = 1:n_C  % 遍历 C 参数
        C_val = svm_C_grid(ci);  % 当前 C 值
        for gi = 1:n_gamma  % 遍历 γ 参数
            gamma_val = svm_gamma_grid(gi);  % 当前 γ 值
            rng(RNG_SEED);  % 固定随机种子
            try
                % 训练 RBF 核 SVM
                svm_tmp = fitcsvm(X_train_norm, y_train, ...
                    'KernelFunction', 'rbf', ...         % RBF（高斯）核函数
                    'BoxConstraint', C_val, ...          % 惩罚参数 C
                    'KernelScale', gamma_val, ...        % 核尺度参数
                    'Standardize', false, ...            % 数据已手动归一化
                    'Verbose', 0);                       % 关闭训练日志
                % 验证集评估
                [y_pred_tmp, ~] = predict(svm_tmp, X_val_norm);
                val_acc = mean(y_pred_tmp == y_val);  % 计算验证准确率
                svm_val_acc_grid(ci, gi) = val_acc;   % 记录
                % 更新最佳模型
                if val_acc > best_svm_acc
                    best_svm_acc = val_acc;
                    best_svm_C = C_val;
                    best_svm_ks = gamma_val;
                    best_svm_mdl = svm_tmp;
                end
                fprintf('    C=%.2f, gamma=%.2f: Val ACC=%.4f\n', C_val, gamma_val, val_acc);
            catch ME
                fprintf('    C=%.2f, gamma=%.2f: 训练失败 (%s)\n', C_val, gamma_val, ME.message);
                svm_val_acc_grid(ci, gi) = 0;  % 标记为失败
            end
        end
    end
    % 记录最优 SVM 参数
    svm_params{s} = struct('BoxConstraint', best_svm_C, 'KernelScale', best_svm_ks);
    fprintf('  -> 最优 SVM: C=%.2f, gamma=%.2f, Val ACC=%.4f\n', ...
        best_svm_C, best_svm_ks, best_svm_acc);
    % 在全部三个数据集上评估最优 SVM
    [y_train_pred_svm, ~] = predict(best_svm_mdl, X_train_norm);
    [y_val_pred_svm,   ~] = predict(best_svm_mdl, X_val_norm);
    [y_test_pred_svm,  ~] = predict(best_svm_mdl, X_test_norm);
    svm_metrics_train = compute_all_metrics(y_train, y_train_pred_svm);
    svm_metrics_val   = compute_all_metrics(y_val,   y_val_pred_svm);
    svm_metrics_test  = compute_all_metrics(y_test,  y_test_pred_svm);
    subject_results{s, 3, 1} = svm_metrics_train;  % SVM 训练集
    subject_results{s, 3, 2} = svm_metrics_val;    % SVM 验证集
    subject_results{s, 3, 3} = svm_metrics_test;   % SVM 测试集
    % 打印当前被试的三个模型测试集结果摘要
    fprintf('  ---- 测试集结果 ----\n');
    fprintf('  LR:   ACC=%.4f, F1=%.4f, Sens=%.4f, Spec=%.4f\n', ...
        lr_metrics_test.ACC, lr_metrics_test.F1, lr_metrics_test.Sens, lr_metrics_test.Spec);
    fprintf('  MLP:  ACC=%.4f, F1=%.4f, Sens=%.4f, Spec=%.4f\n', ...
        mlp_metrics_test.ACC, mlp_metrics_test.F1, mlp_metrics_test.Sens, mlp_metrics_test.Spec);
    fprintf('  SVM:  ACC=%.4f, F1=%.4f, Sens=%.4f, Spec=%.4f\n', ...
        svm_metrics_test.ACC, svm_metrics_test.F1, svm_metrics_test.Sens, svm_metrics_test.Spec);
end
%% ===== 5. 汇总统计：计算各指标在被试群体的均值与标准差 =====
fprintf('\n========== 汇总统计结果 ==========\n');
% 统计各模型、各数据集划分、各指标在被试群体上的均值和标准差
% summary_stats{m, sp}.metric_mean(i) = mean
% summary_stats{m, sp}.metric_std(i)  = std
summary_stats = struct();  % 初始化汇总结构体
for m = 1:n_models  % 遍历模型
    for sp = 1:n_splits  % 遍历数据集划分
        % 初始化指标收集矩阵 [被试数 × 指标数]
        all_metrics_mat = [];  % 用于收集有效被试的指标
        valid_subj_count = 0;  % 有效被试计数
        for s = 1:n_subjects
            if ~isempty(subject_results{s, m, sp})
                metrics_s = subject_results{s, m, sp};  % 当前被试的指标结构体
                valid_subj_count = valid_subj_count + 1;
                % 按顺序提取 6 个指标
                all_metrics_mat(valid_subj_count, :) = [
                    metrics_s.ACC, metrics_s.F1, metrics_s.Sens, ...
                    metrics_s.Spec, metrics_s.Prec, metrics_s.NPV];
            end
        end
        % 计算均值和标准差
        summary_stats(m, sp).mean_vals = mean(all_metrics_mat, 1);  % 各指标均值
        summary_stats(m, sp).std_vals  = std(all_metrics_mat, 0, 1);  % 各指标标准差
        summary_stats(m, sp).n_valid    = valid_subj_count;  % 有效被试数
    end
end
% ----- 5a. 输出每名被试的独立分类结果（测试集） -----
fprintf('\n========== 各被试测试集独立结果 ==========\n');
for m = 1:n_models  % 遍历模型
    fprintf('\n--- 模型: %s ---\n', model_names{m});
    % 打印表头
    fprintf('被试\tACC\tF1\tSens\tSpec\tPrec\tNPV\n');
    for s = 1:n_subjects
        if ~isempty(subject_results{s, m, 3})  % 测试集 = 划分 3
            mt = subject_results{s, m, 3};
            fprintf('S%02d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n', ...
                subj_ids(s), mt.ACC, mt.F1, mt.Sens, mt.Spec, mt.Prec, mt.NPV);
        else
            fprintf('S%02d\t-\t-\t-\t-\t-\t-\n', subj_ids(s));
        end
    end
    % 打印均值±标准差
    fprintf('Mean\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n', summary_stats(m, 3).mean_vals);
    fprintf('Std\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n', summary_stats(m, 3).std_vals);
end
% ----- 5b. 输出全部被试汇总结果（三个数据集上的均值±标准差） -----
fprintf('\n========== 全部被试汇总结果（均值±标准差） ==========\n');
for sp = 1:n_splits  % 遍历数据集划分
    fprintf('\n===== %s =====\n', split_names{sp});
    % 打印表头
    fprintf('%-25s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
        '模型', 'ACC', 'F1', 'Sens', 'Spec', 'Prec', 'NPV');
    fprintf('%-25s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
        '', '(M±SD)', '(M±SD)', '(M±SD)', '(M±SD)', '(M±SD)', '(M±SD)');

    for m = 1:n_models  % 遍历模型
        mv = summary_stats(m, sp).mean_vals;  % 当前模型各指标均值
        sv = summary_stats(m, sp).std_vals;   % 当前模型各指标标准差
        fprintf('%-25s %5.3f±%5.3f %5.3f±%5.3f %5.3f±%5.3f %5.3f±%5.3f %5.3f±%5.3f %5.3f±%5.3f\n', ...
            model_names{m}, mv(1), sv(1), mv(2), sv(2), mv(3), sv(3), mv(4), sv(4), mv(5), sv(5), mv(6), sv(6));
    end
end
%% ===== 6. 保存结果 =====
fprintf('\n========== 保存结果文件 ==========\n');

result_file = fullfile(res_dir, 'SubjectDependent_Results.mat');
save(result_file, 'subject_results', 'summary_stats', 'mlp_architectures', 'svm_params', ...
    'SELECTED_CONFIG', 'SELECTED_FEATURES', 'SPLIT_RATIO', 'model_names', 'metric_names', ...
    'split_names', 'subj_ids', 'n_subjects', '-v7.3');
fprintf('结果已保存至: %s\n', result_file);

% 导出 CSV 表格
export_results_csv(res_dir, 'SubjectDependent', subj_ids, subject_results, ...
    model_names, metric_names, split_names, n_subjects, n_models, n_splits, n_metrics);

% 打印 MLP 架构选择统计
fprintf('\n========== MLP 架构选择统计 ==========\n');
arch_counts = containers.Map();  % 统计每种架构被选中的次数
for s = 1:n_subjects
    if ~isempty(mlp_architectures{s})
        if isempty(mlp_architectures{s})
            arch_key = '线性';  % 无隐藏层
        else
            arch_key = mat2str(mlp_architectures{s});  % 转为字符串
        end
        if isKey(arch_counts, arch_key)
            arch_counts(arch_key) = arch_counts(arch_key) + 1;
        else
            arch_counts(arch_key) = 1;
        end
    end
end
all_arch_keys = keys(arch_counts);  % 所有架构类型
for i = 1:length(all_arch_keys)
    fprintf('  %s: %d 名被试\n', all_arch_keys{i}, arch_counts(all_arch_keys{i}));
end
fprintf('\n========== 被试依赖分类实验全部完成 ==========\n');
%% ===== 辅助函数 =====
function metrics = compute_all_metrics(y_true, y_pred)
    % 计算二分类全部 6 项评价指标
    % 输入:
    %   y_true - 真实标签向量 (0=负类/低负荷, 1=正类/高负荷)
    %   y_pred - 预测标签向量
    % 输出:
    %   metrics - 包含 ACC, F1, Sens, Spec, Prec, NPV 的结构体
    % 计算混淆矩阵元素
    TP = sum(y_true == 1 & y_pred == 1);  % 真阳性：实际高负荷且预测高负荷
    TN = sum(y_true == 0 & y_pred == 0);  % 真阴性：实际低负荷且预测低负荷
    FP = sum(y_true == 0 & y_pred == 1);  % 假阳性：实际低负荷但预测高负荷
    FN = sum(y_true == 1 & y_pred == 0);  % 假阴性：实际高负荷但预测低负荷
    total = TP + TN + FP + FN;  % 样本总数
    % ACC: 准确率 = (TP+TN) / 总数
    if total > 0
        metrics.ACC = (TP + TN) / total;
    else
        metrics.ACC = 0;
    end
    % Sens: 灵敏度/召回率 = TP / (TP+FN)
    if (TP + FN) > 0
        metrics.Sens = TP / (TP + FN);
    else
        metrics.Sens = 0;
    end
    % Spec: 特异度 = TN / (TN+FP)
    if (TN + FP) > 0
        metrics.Spec = TN / (TN + FP);
    else
        metrics.Spec = 0;
    end
    % Prec: 精确率 = TP / (TP+FP)
    if (TP + FP) > 0
        metrics.Prec = TP / (TP + FP);
    else
        metrics.Prec = 0;
    end
    % NPV: 阴性预测值 = TN / (TN+FN)
    if (TN + FN) > 0
        metrics.NPV = TN / (TN + FN);
    else
        metrics.NPV = 0;
    end
    % F1: F1 分数 = 2 * Prec * Sens / (Prec + Sens)
    if (metrics.Prec + metrics.Sens) > 0
        metrics.F1 = 2 * metrics.Prec * metrics.Sens / (metrics.Prec + metrics.Sens);
    else
        metrics.F1 = 0;
    end
end

function export_results_csv(res_dir, prefix, subj_ids, subject_results, ...
    model_names, metric_names, split_names, n_subjects, n_models, n_splits, n_metrics)
    % 导出分类结果为 CSV 文件
    % 为每个数据集划分创建独立的 CSV 文件
    for sp = 1:n_splits  % 遍历数据集划分
        csv_rows = {};  % CSV 行数据
        row_idx = 0;  % 行计数器

        for m = 1:n_models  % 遍历模型
            for s = 1:n_subjects  % 遍历被试
                if ~isempty(subject_results{s, m, sp})
                    row_idx = row_idx + 1;
                    mt = subject_results{s, m, sp};
                    csv_rows{row_idx, 1} = sprintf('S%02d', subj_ids(s));  % 被试编号
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

        % 写入 CSV 文件
        if ~isempty(csv_rows)
            csv_table = cell2table(csv_rows, 'VariableNames', ...
                {'Subject', 'Model', 'ACC', 'F1', 'Sensitivity', 'Specificity', 'Precision', 'NPV'});
            csv_file = fullfile(res_dir, sprintf('%s_%s.csv', prefix, split_names{sp}));
            writetable(csv_table, csv_file);
            fprintf('已导出: %s\n', csv_file);
        end
    end
end

function s = conditional(cond, val_true, val_false)
    % 简易三元条件运算符
    if cond
        s = val_true;
    else
        s = val_false;
    end
end
