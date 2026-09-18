clear; clc; close all;
%% ===== 1. 加载汇总特征数据 =====
feat_dir = fullfile(fileparts(mfilename('fullpath')), 'features');
fig_dir  = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
% 加载汇总数据文件
summary_file = fullfile(feat_dir, 'AllFeatures_Summary.mat');
if ~exist(summary_file, 'file')
    error('汇总特征文件不存在: %s\n请先运行 b1_segmentation_feature_extraction.m', summary_file);
end
load(summary_file);
fprintf('已加载汇总特征数据\n');
n_bands = length(band_names);  % 频段数量
%% ===== 2. 配置可视化参数 =====
% 选择需要可视化的特征类型索引
% 此处选择全部8种特征类型进行可视化
viz_feature_indices = 1:8;  % 1=PSD, 2=DE, 3=LEE, 4=ZCR, 5=Kurtosis, 6=Skewness, 7=Mean, 8=Variance
% 选择需要可视化的分段配置索引（1-6）
% 1=1s_noOverlap, 2=1s_50pOverlap, 3=2s_noOverlap, 4=2s_50pOverlap, 5=4s_noOverlap, 6=4s_50pOverlap
viz_config_indices = 1:6;
% 频段对应的颜色映射（用于PSD/DE/LEE等频段特征）
band_colors = {'#0072BD', '#D95319', '#EDB120', '#7E2F8E'};
band_labels_display = {'\theta (4-8 Hz)', '\alpha (8-13 Hz)', '\beta (13-30 Hz)', '\gamma (30-45 Hz)'};
% 条件颜色：lo=蓝色, hi=红色 (RGB三元组格式)
cond_colors = [0 0.4470 0.7410;   % 蓝色 (lo)
               0.8500 0.3250 0.0980];  % 红色 (hi)
cond_color_names = {'b', 'r'};  % 散点图用颜色字符
cond_labels_display = {'低负荷(静息)', '高负荷(心算)'};
%% ===== 3. 特征可视化：被试全体箱型图 =====
% 对每种分段配置、每种特征类型，绘制lo vs hi的箱型图
fprintf('\n========== 生成被试全体箱型图 ==========\n');
for c = viz_config_indices
    cfg = seg_configs(c);
    cfg_name = cfg.name;
    for f = viz_feature_indices
        feat_name = feature_type_names{f};
        % 获取当前配置+特征类型的lo和hi数据
        feat_lo = pooled_features{c}.lo{f};
        feat_hi = pooled_features{c}.hi{f};
        if isempty(feat_lo) || isempty(feat_hi)
            fprintf('  跳过: %s - %s (数据为空)\n', cfg_name, feat_name);
            continue;
        end
        % 确定子特征数量
        n_subfeat = size(feat_lo, 2);
        % 创建图像
        fig = figure('Visible', 'off', 'Position', [50, 50, 200 * n_subfeat + 200, 600]);
        fig.Name = sprintf('全体箱型图: %s - %s', cfg_name, feat_name);
        for sf = 1:n_subfeat
            subplot(1, n_subfeat, sf);
            % 将lo和hi数据组合绘制箱型图
            box_data = [feat_lo(:, sf); feat_hi(:, sf)];
            box_group = [ones(size(feat_lo(:, sf))); 2 * ones(size(feat_hi(:, sf)))];
            boxplot(box_data, box_group, 'Colors', cond_colors, 'Symbol', '.');
            hold on;
            % 绘制个体数据散点（随机抽样减少密度）
            n_lo = size(feat_lo, 1);
            n_hi = size(feat_hi, 1);
            max_scatter = 500;  % 每个条件最多显示500个散点
            rng(42);  % 固定随机种子保证可重复性
            % lo散点
            idx_lo = randperm(n_lo, min(max_scatter, n_lo));
            scatter(ones(length(idx_lo),1) + (rand(length(idx_lo),1)-0.5)*0.3, ...
                feat_lo(idx_lo, sf), 4, cond_colors(1,:), 'filled', 'MarkerFaceAlpha', 0.3);
            % hi散点
            idx_hi = randperm(n_hi, min(max_scatter, n_hi));
            scatter(2*ones(length(idx_hi),1) + (rand(length(idx_hi),1)-0.5)*0.3, ...
                feat_hi(idx_hi, sf), 4, cond_colors(2,:), 'filled', 'MarkerFaceAlpha', 0.3);
            hold off;
            % 设置子图标签
            set(gca, 'XTickLabel', cond_labels_display);
            if n_subfeat > 1 && ismember(f, [1,2,3])
                % PSD/DE/LEE特征有4个频段子特征
                title(sprintf('%s', band_labels_display{sf}), 'Interpreter', 'tex');
            end
            ylabel('特征值');
            grid on;
        end
        % 设置总标题
        sgtitle(sprintf('特征: %s | 分段: %s (窗长%ds, %s重叠) | 全体被试 (n=%d)', ...
            feat_name, cfg_name, cfg.win_len_sec, ...
            conditional(cfg.overlap_ratio == 0, '无', '50%'), n_subjects), ...
            'Interpreter', 'none');
        % 保存图像
        fig_file = fullfile(fig_dir, sprintf('Boxplot_AllSubj_%s_%s.png', cfg_name, feat_name));
        saveas(fig, fig_file);
        close(fig);
        fprintf('  已保存: %s\n', fig_file);
    end
end
%% ===== 4. 特征可视化：被试依赖箱型图 =====
% 对每种分段配置、每种特征类型，绘制每个被试的lo vs hi箱型图
fprintf('\n========== 生成被试依赖箱型图 ==========\n');
for c = viz_config_indices
    cfg = seg_configs(c);
    cfg_name = cfg.name;
    for f = viz_feature_indices
        feat_name = feature_type_names{f};

        % 检查是否有被试数据
        has_data = false;
        for s = 1:n_subjects
            if ~isempty(subj_features{c, s})
                has_data = true;
                break;
            end
        end
        if ~has_data
            fprintf('  跳过: %s - %s (无被试数据)\n', cfg_name, feat_name);
            continue;
        end
        n_subfeat_vals = [];  % 确定子特征数
        for s = 1:n_subjects
            if ~isempty(subj_features{c, s}) && isfield(subj_features{c, s}, 'lo_feats')
                n_subfeat_vals = size(subj_features{c, s}.lo_feats{f}, 3);
                break;
            end
        end
        if isempty(n_subfeat_vals)
            n_subfeat_vals = size(subj_features{c, s}.hi_feats{f}, 3);
        end
        if isempty(n_subfeat_vals)
            n_subfeat_vals = 1;
        end
        % 对每个子特征，为所有被试绘制箱型图
        for sf = 1:n_subfeat_vals
            fig = figure('Visible', 'off', ...
                'Position', [50, 50, min(1800, 80 * n_subjects + 200), 700]);
            fig.Name = sprintf('被试依赖箱型图: %s - %s - 子特征%d', cfg_name, feat_name, sf);
            % 为每个被试收集lo和hi数据
            lo_data_by_subj = cell(n_subjects, 1);
            hi_data_by_subj = cell(n_subjects, 1);
            subj_labels = cell(n_subjects, 1);
            for s = 1:n_subjects
                subj_labels{s} = sprintf('S%02d', subj_ids(s));
                if ~isempty(subj_features{c, s})
                    if isfield(subj_features{c, s}, 'lo_feats')
                        feat_s_lo = subj_features{c, s}.lo_feats{f};
                        % feat维度: [nSeg × nCh × nSubFeat]
                        feat_s_lo_2d = reshape(feat_s_lo(:, :, sf), [], 1);
                        lo_data_by_subj{s} = feat_s_lo_2d;
                    else
                        lo_data_by_subj{s} = [];
                    end
                    if isfield(subj_features{c, s}, 'hi_feats')
                        feat_s_hi = subj_features{c, s}.hi_feats{f};
                        feat_s_hi_2d = reshape(feat_s_hi(:, :, sf), [], 1);
                        hi_data_by_subj{s} = feat_s_hi_2d;
                    else
                        hi_data_by_subj{s} = [];
                    end
                else
                    lo_data_by_subj{s} = [];
                    hi_data_by_subj{s} = [];
                end
            end
            % 使用分组箱型图绘制
            % 每个被试有2个箱子（lo和hi）
            n_s = n_subjects;
            box_data_all = [];
            box_group_all = [];
            box_positions = [];
            for s = 1:n_s
                if ~isempty(lo_data_by_subj{s})
                    box_data_all = [box_data_all; lo_data_by_subj{s}];
                    box_group_all = [box_group_all; 2*s-1 * ones(length(lo_data_by_subj{s}), 1)];
                end
                if ~isempty(hi_data_by_subj{s})
                    box_data_all = [box_data_all; hi_data_by_subj{s}];
                    box_group_all = [box_group_all; 2*s * ones(length(hi_data_by_subj{s}), 1)];
                end
            end
            if ~isempty(box_data_all)
                % 计算位置
                positions = 1:(2*n_s);
                % 构建2*n_s的RGB矩阵（交替lo蓝/hi红）
                box_cols_rgb = repmat(cond_colors, n_s, 1);
                boxplot(box_data_all, box_group_all, 'Positions', positions, ...
                    'Colors', box_cols_rgb, 'Symbol', '.');
                hold on;
                % 添加分隔线和被试标签
                for s = 1:n_s
                    x_center = 2*s - 0.5;
                    xline(x_center, ':', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
                end
                hold off;
                % 设置X轴标签
                xtick_positions = 1.5:2:(2*n_s);
                set(gca, 'XTick', xtick_positions, 'XTickLabel', subj_labels, ...
                    'XTickLabelRotation', 45);
            end
            ylabel('特征值');
            grid on;
            % 设置标题
            if n_subfeat_vals > 1 && ismember(f, [1,2,3])
                feat_title = sprintf('%s - %s', feat_name, band_labels_display{sf});
            else
                feat_title = feat_name;
            end
            sgtitle(sprintf('特征: %s | 分段: %s (窗长%ds, %s重叠) | 每被试箱型图', ...
                feat_title, cfg_name, cfg.win_len_sec, ...
                conditional(cfg.overlap_ratio == 0, '无', '50%')), ...
                'Interpreter', 'none');
            % 保存图像
            fig_file = fullfile(fig_dir, sprintf('Boxplot_BySubj_%s_%s_sf%d.png', ...
                cfg_name, feat_name, sf));
            saveas(fig, fig_file);
            close(fig);
        end
        fprintf('  已保存: %s - %s (被试依赖)\n', cfg_name, feat_name);
    end
end
%% ===== 5. 双侧T检验：评估不同认知负荷下的特征均值差异 =====
fprintf('\n========== 双侧T检验分析 ==========\n');
% 使用cell数组收集结果（比动态扩增table更高效）
tt_rows = {};  % 每行存储一个检验结果
result_idx = 0;
for c = viz_config_indices
    cfg = seg_configs(c);
    cfg_name = cfg.name;
    for f = viz_feature_indices
        feat_name = feature_type_names{f};
        feat_lo = pooled_features{c}.lo{f};
        feat_hi = pooled_features{c}.hi{f};
        if isempty(feat_lo) || isempty(feat_hi)
            continue;
        end
        n_subfeat = size(feat_lo, 2);
        for sf = 1:n_subfeat
            % 提取lo和hi条件的特征值
            data_lo = feat_lo(:, sf);
            data_hi = feat_hi(:, sf);
            % 去除可能的NaN和Inf值
            valid_idx_lo = isfinite(data_lo);
            valid_idx_hi = isfinite(data_hi);
            data_lo_clean = data_lo(valid_idx_lo);
            data_hi_clean = data_hi(valid_idx_hi);
            if length(data_lo_clean) < 3 || length(data_hi_clean) < 3
                continue;  % 样本数不足
            end
            % 执行双侧T检验（独立样本，不等方差Welch校正）
            [h, p_val, ~, stats] = ttest2(data_lo_clean, data_hi_clean, ...
                'Vartype', 'unequal', 'Tail', 'both');
            % 计算Cohen's d效应量
            pooled_std = sqrt((var(data_lo_clean) + var(data_hi_clean)) / 2);
            if pooled_std > 0
                cohens_d = (mean(data_hi_clean) - mean(data_lo_clean)) / pooled_std;
            else
                cohens_d = 0;
            end
            % 子特征名称确定
            if ismember(f, [1,2,3]) && n_subfeat > 1
                subfeat_label = band_names{sf};
            else
                subfeat_label = 'N/A';
            end
            % 记录结果到cell数组
            result_idx = result_idx + 1;
            tt_rows{result_idx, 1} = cfg_name;
            tt_rows{result_idx, 2} = feat_name;
            tt_rows{result_idx, 3} = subfeat_label;
            tt_rows{result_idx, 4} = mean(data_lo_clean);
            tt_rows{result_idx, 5} = mean(data_hi_clean);
            tt_rows{result_idx, 6} = std(data_lo_clean);
            tt_rows{result_idx, 7} = std(data_hi_clean);
            tt_rows{result_idx, 8} = length(data_lo_clean);
            tt_rows{result_idx, 9} = length(data_hi_clean);
            tt_rows{result_idx, 10} = stats.tstat;
            tt_rows{result_idx, 11} = stats.df;
            tt_rows{result_idx, 12} = p_val;
            tt_rows{result_idx, 13} = cohens_d;
            tt_rows{result_idx, 14} = h;
            tt_rows{result_idx, 15} = get_sig_level(p_val);
            % 打印结果
            sig_marker = '';
            if h == 1
                sig_marker = ' *** 显著 ***';
            end
            fprintf('  [%s] %s-%s: t=%.3f, df=%.1f, p=%.4f, d=%.3f | lo_mean=%.4f, hi_mean=%.4f%s\n', ...
                cfg_name, feat_name, subfeat_label, ...
                stats.tstat, stats.df, p_val, cohens_d, ...
                mean(data_lo_clean), mean(data_hi_clean), sig_marker);
        end
    end
end
% 将cell数组转换为table
t_test_results = cell2table(tt_rows, 'VariableNames', {
    'config_name', 'feature_type', 'sub_feature', ...
    'mean_lo', 'mean_hi', 'std_lo', 'std_hi', ...
    'n_lo', 'n_hi', 't_statistic', 'df', ...
    'p_value', 'cohens_d', 'significant', 'sig_level'});
% 保存T检验结果表
t_test_file = fullfile(feat_dir, 'TTest_Results.mat');
save(t_test_file, 't_test_results', '-v7.3');
% 导出CSV格式
t_test_csv = fullfile(fig_dir, 'TTest_Results.csv');
writetable(t_test_results, t_test_csv);
fprintf('\nT检验结果已保存至: %s\n', t_test_file);
fprintf('T检验结果CSV: %s\n', t_test_csv);
%% ===== 6. 显著性结果可视化 =====
fprintf('\n========== 生成显著性热力图 ==========\n');
% 6a. 为每种分段配置绘制显著性热力图（特征×频段）
for c = viz_config_indices
    cfg = seg_configs(c);
    cfg_name = cfg.name;
    % 构建p值矩阵: [8特征 × 4频段]
    % 特征1-3有4个频段子特征，特征4-8只有1个子特征
    p_matrix = nan(n_feature_types, n_bands);
    d_matrix = nan(n_feature_types, n_bands);  % Cohen's d矩阵
    for f = 1:n_feature_types
        feat_lo = pooled_features{c}.lo{f};
        feat_hi = pooled_features{c}.hi{f};
        if isempty(feat_lo), continue; end
        n_sf = size(feat_lo, 2);
        for sf = 1:n_sf
            data_lo = feat_lo(isfinite(feat_lo(:, sf)), sf);
            data_hi = feat_hi(isfinite(feat_hi(:, sf)), sf);
            if length(data_lo) < 3 || length(data_hi) < 3, continue; end

            [~, p_val, ~, stats] = ttest2(data_lo, data_hi, 'Vartype', 'unequal');
            pooled_std = sqrt((var(data_lo) + var(data_hi)) / 2);
            if pooled_std > 0
                d_val = (mean(data_hi) - mean(data_lo)) / pooled_std;
            else
                d_val = 0;
            end
            if n_sf == n_bands
                p_matrix(f, sf) = p_val;
                d_matrix(f, sf) = d_val;
            else
                p_matrix(f, 1) = p_val;
                d_matrix(f, 1) = d_val;
            end
        end
    end
    % 绘制P值热力图
    fig = figure('Visible', 'off', 'Position', [100, 100, 900, 500]);
    subplot(1, 2, 1);
    imagesc(-log10(p_matrix));
    colormap(jet);
    colorbar;
    set(gca, 'XTick', 1:n_bands, 'XTickLabel', band_labels_display);
    set(gca, 'YTick', 1:n_feature_types, 'YTickLabel', feature_type_names);
    title(sprintf('P值热力图 (-log10(p)): %s', cfg_name), 'Interpreter', 'none');
    xlabel('频段');
    ylabel('特征类型');
    % 标注显著性
    for i = 1:n_feature_types
        for j = 1:n_bands
            if ~isnan(p_matrix(i, j))
                sig_str = get_sig_level(p_matrix(i, j));
                if ~strcmp(sig_str, 'ns')
                    text(j, i, sig_str, 'HorizontalAlignment', 'center', ...
                        'FontWeight', 'bold', 'FontSize', 8, 'Color', 'white');
                end
            end
        end
    end
    % 绘制Cohen's d热力图
    subplot(1, 2, 2);
    imagesc(d_matrix);
    colormap(redbluecmap);
    colorbar;
    set(gca, 'XTick', 1:n_bands, 'XTickLabel', band_labels_display);
    set(gca, 'YTick', 1:n_feature_types, 'YTickLabel', feature_type_names);
    title(sprintf('效应量 (Cohen''s d): %s', cfg_name), 'Interpreter', 'none');
    xlabel('频段');
    ylabel('特征类型');
    % 标注效应量数值
    for i = 1:n_feature_types
        for j = 1:n_bands
            if ~isnan(d_matrix(i, j))
                text(j, i, sprintf('%.2f', d_matrix(i, j)), ...
                    'HorizontalAlignment', 'center', 'FontSize', 7);
            end
        end
    end
    sgtitle(sprintf('分段配置: %s (窗长%ds, %s重叠)\n双侧T检验显著性分析', ...
        cfg_name, cfg.win_len_sec, conditional(cfg.overlap_ratio == 0, '无', '50%')), ...
        'Interpreter', 'none');

    fig_file = fullfile(fig_dir, sprintf('SignificanceHeatmap_%s.png', cfg_name));
    saveas(fig, fig_file);
    close(fig);
    fprintf('  已保存显著性热力图: %s\n', fig_file);
end
%% ===== 7. 显著性标记的汇总条形图 =====
fprintf('\n========== 生成显著性汇总条形图 ==========\n');
% 统计每种分段配置下达到显著(p<0.05)的特征数量
sig_counts = zeros(n_configs, n_feature_types);
for c = 1:n_configs
    for f = 1:n_feature_types
        feat_lo_all = pooled_features{c}.lo{f};
        feat_hi_all = pooled_features{c}.hi{f};
        if isempty(feat_lo_all), continue; end
        n_sf = size(feat_lo_all, 2);
        sig_count = 0;
        for sf = 1:n_sf
            data_lo = feat_lo_all(isfinite(feat_lo_all(:, sf)), sf);
            data_hi = feat_hi_all(isfinite(feat_hi_all(:, sf)), sf);
            if length(data_lo) < 3 || length(data_hi) < 3, continue; end
            [h, ~] = ttest2(data_lo, data_hi, 'Vartype', 'unequal');
            sig_count = sig_count + h;
        end
        sig_counts(c, f) = sig_count;
    end
end
fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 600]);
b = bar(sig_counts');
set(gca, 'XTickLabel', feature_type_names, 'XTickLabelRotation', 30);
legend(arrayfun(@(x) x.name, seg_configs, 'UniformOutput', false), ...
    'Location', 'best', 'Interpreter', 'none');
ylabel('显著子特征数量');
title('各分段配置下达到显著差异(p<0.05)的子特征数量');
grid on;
fig_file = fullfile(fig_dir, 'SignificanceSummary_Bar.png');
saveas(fig, fig_file);
close(fig);
fprintf('  已保存显著性汇总条形图: %s\n', fig_file);

fprintf('\n========== 可视化与统计分析全部完成 ==========\n');
%% ===== 辅助函数 =====
function sig_str = get_sig_level(p)
    % 根据p值返回显著性标记字符串
    if p < 0.001
        sig_str = '***';
    elseif p < 0.01
        sig_str = '**';
    elseif p < 0.05
        sig_str = '*';
    else
        sig_str = 'ns';
    end
end
function cm = redbluecmap()
    % 生成红白色系colormap用于效应量
    n = 256;
    r = [(0:n-1)'/(n-1), zeros(n, 1), (n-1:-1:0)'/(n-1)];
    cm = r;
end
function s = conditional(cond, val_true, val_false)
    % 简易条件函数（MATLAB无内置三元运算符）
    if cond
        s = val_true;
    else
        s = val_false;
    end
end
