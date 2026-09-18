clear;
clc; 
close all;
eeglab; % 启动EEGLAB工具箱，自动加载已安装插件
data_dir = fullfile(fileparts(mfilename('fullpath')),'eeg-during-mental-arithmetic-tasks-1.0.0','eeg-during-mental-arithmetic-tasks-1.0.0'); % 拼接原始EDF数据存储路径
output_dir = fullfile(fileparts(mfilename('fullpath')), 'preprocessed_data'); % 拼接预处理结果保存路径
if ~exist(output_dir, 'dir') % 判断结果文件夹是否不存在
    mkdir(output_dir); % 新建预处理结果存储文件夹
end
FS = 500; % 定义原始EEG信号采样频率，单位Hz
N_CHANNELS = 19; % 定义实验有效脑电导联总数量
N_SAMPLES = 30000; % 定义单条原始数据采样总点数
EPOCH_DURATION = 2; % 定义单次分段Epoch的时间长度，单位s
EPOCH_OVERLAP = 0; % 定义Epoch分段重叠时长
chan_labels = {'Fp1','Fp2','F3','F4','F7','F8','T3','T4','C3','C4','T5','T6','P3','P4','O1','O2','Fz','Cz','Pz'}; % 录入国际10-20系统19个电极标签
bands = struct('delta', [0.5 4],'theta', [4 8],'alpha', [8 13],'beta',  [13 30],'gamma', [30 45]); % 利用结构体存储五种脑电节律的频率范围
subject_range = 0:35; % 固定范围，代码自动跳过已处理完成的被试
condition_tags = {'lo', 'hi'}; % 定义任务标识，lo为静息无负荷，hi为高负荷心算
edf_suffixes  = {'_1', '_2'}; % 定义EDF文件后缀，_1对应静息，_2对应心算任务

for sub = subject_range % 外层循环，逐个遍历所有被试编号
    for cond = 1:2 % 内层循环，依次处理静息、心算两种实验条件
        % 拼接输出文件名，用于判断是否已经处理完毕
        out_name = sprintf('Sub%02d_%s_preprocessed.mat', sub, condition_tags{cond});
        out_full = fullfile(output_dir, out_name);
        % 新增：文件已存在则直接跳过，不用重复运算
        if exist(out_full, 'file')
            fprintf('\n===== 被试%d-%s 已存在，跳过处理 =====\n',sub,condition_tags{cond});
            continue;
        end

        edf_filename = sprintf('Subject%02d%s.edf', sub, edf_suffixes{cond}); % 格式化生成单个被试单任务EDF文件名
        edf_fullpath = fullfile(data_dir, edf_filename); % 拼接得到EDF文件完整绝对路径
        fprintf('\n========================================\n'); % 打印分隔符
        fprintf('正在读取: %s\n', edf_filename); % 输出当前读取的文件名称
        fprintf('========================================\n'); % 打印分隔符
        [hdr, ~] = edfread(edf_fullpath); % MATLAB内置函数读取EDF文件，返回表格格式数据
        x    = table2array(hdr); % 将表格类型数据转换为元胞数组
        eegb = cell2mat(x); % 元胞数组转为纯数值二维矩阵
        eeg_raw = eegb(1:N_SAMPLES, 1:N_CHANNELS)'; % 截取有效数据并转置为【导联×采样点】格式
        clear hdr x eegb; % 清除临时中间变量，释放内存
        fprintf('原始数据维度: %d 导联 × %d 采样点\n', size(eeg_raw, 1), size(eeg_raw, 2)); % 输出原始数据矩阵尺寸
        EEG = pop_importdata('dataformat', 'array','data', eeg_raw,'srate', FS,'chanlocs', [],'setname', sprintf('Sub%02d_%s_raw', sub, condition_tags{cond})); % 将数值矩阵导入EEGLAB标准EEG结构体
        for ch = 1:N_CHANNELS % 循环遍历所有导联
            EEG.chanlocs(ch).labels = chan_labels{ch}; % 为每个导联结构体赋值电极名称
        end
        EEG.nbchan = N_CHANNELS; % 向EEG结构体写入总导联数目
        EEG = pop_chanedit(EEG, 'lookup',fullfile(fileparts(which('eeglab')), 'plugins','dipfit', 'standard_BESA', 'standard-10-5-cap385.elp')); % 加载标准电极坐标文件，自动匹配导联三维位置
        unmatched = find(cellfun(@(s) isempty(s.X), num2cell(EEG.chanlocs))); % 查找未匹配到空间坐标的电极索引
        if ~isempty(unmatched) % 判定存在坐标未匹配的导联
            fprintf('警告: 以下导联未匹配到标准坐标，将使用默认值:\n'); % 输出警告提示
            for u = unmatched % 遍历异常导联编号
                fprintf('  - %s\n', EEG.chanlocs(u).labels); % 打印未匹配坐标的电极名称
            end
        end
        EEG = eeg_checkset(EEG); % EEGLAB内置函数校验EEG结构体格式合规性
        fprintf('通道位置加载完成，共 %d 个导联\n', EEG.nbchan); % 输出导联加载结果
        fprintf('参考电极: %s\n', EEG.ref); % 输出原始数据参考类型
        fprintf('\n--- 执行 50 Hz 陷波滤波 ---\n'); % 打印滤波步骤提示
        notch_freq = 50; % 设置工频干扰中心频率50Hz
        notch_bw   = 1; % 设置陷波滤波器通带带宽
        EEG = pop_eegfiltnew(EEG,'locutoff', notch_freq - notch_bw/2,'hicutoff', notch_freq + notch_bw/2,'revfilt', 1,'plotfreqz', 0); % 带阻滤波去除50Hz工频噪声，关闭滤波器绘图
        EEG.setname = sprintf('%s_notch', EEG.setname); % 修改数据集名称，标记已完成陷波
        EEG = eeg_checkset(EEG); % 校验滤波后数据结构体
        fprintf('50 Hz 陷波滤波完成 (阻带: %.1f ~ %.1f Hz)\n',notch_freq - notch_bw/2, notch_freq + notch_bw/2); % 输出陷波滤波频率范围
        fprintf('\n--- 执行带通滤波 (0.5 ~ 45 Hz) ---\n'); % 打印带通滤波提示
        bp_low  = 0.5; % 设定带通滤波低频截止频率
        bp_high = 45; % 设定带通滤波高频截止频率
        EEG = pop_eegfiltnew(EEG,'locutoff', bp_low,'hicutoff', bp_high,'plotfreqz', 0); % 全频段带通滤波，去除基线漂移与高频噪声
        EEG.setname = sprintf('%s_bp', EEG.setname); % 更新数据集名称标记带通处理
        EEG = eeg_checkset(EEG); % 校验带通滤波后数据
        fprintf('带通滤波完成: %.1f ~ %.1f Hz (通带)\n', bp_low, bp_high); % 输出带通滤波频段
        fprintf('\n--- 执行重参考: 转为平均参考 ---\n'); % 打印重参考步骤提示
        EEG_preRef = EEG; % 备份重参考之前的原始EEG数据
        EEG = pop_reref(EEG, []); % 全导联求取均值实现平均重参考
        EEG.setname = sprintf('%s_avref', EEG.setname); % 命名标记平均重参考处理
        EEG = eeg_checkset(EEG); % 校验重参考后数据
        fprintf('重参考完成: 已转为平均参考 (共 %d 导联参与)\n', EEG.nbchan); % 输出重参考完成信息
        fprintf('\n--- 执行 ICA 分解 (可能耗时较长) ---\n'); % 打印ICA分解提示
        n_ica_components = N_CHANNELS - 1; % 设定ICA分解成分数，导联数减一
        EEG = pop_runica(EEG,'icatype', 'runica','extended', 1,'pca', n_ica_components,'verbose', 'off'); % 采用扩展Infomax算法完成ICA分解，关闭迭代日志
        EEG.setname = sprintf('%s_ica', EEG.setname); % 命名标记ICA分解完毕
        EEG = eeg_checkset(EEG); % 校验ICA处理后结构体
        fprintf('ICA 分解完成: 提取了 %d 个独立成分\n', size(EEG.icaweights, 1)); % 输出独立成分数量
        fprintf('\n--- ICLabel 自动成分分类 ---\n'); % 打印ICLabel分类提示
        EEG = iclabel(EEG, 'default'); % 调用ICLabel预训练深度学习模型自动分类独立成分
        fprintf('\nICLabel 分类结果 (前 5 个成分预览):\n'); % 打印预览表头
        fprintf('IC# | Brain%% | Muscle%% | Eye%% | Heart%% | LineNoise%% | ChanNoise%% | Other%%\n'); % 打印分类字段标题
        for ic = 1:min(5, size(EEG.icaact, 1)) % 循环选取前5个独立成分
            fprintf('%3d | %6.1f | %7.1f | %5.1f | %6.1f | %9.1f | %9.1f | %6.1f\n',ic, EEG.etc.ic_classification.ICLabel.classifications(ic, :) * 100); % 打印各成分分类概率
        end
        auto_threshold = 0.80; % 设置伪迹自动剔除判定阈值80%
        auto_reject_mask = false(1, size(EEG.icaweights, 1)); % 初始化伪迹标记逻辑数组
        classifications = EEG.etc.ic_classification.ICLabel.classifications; % 提取全部独立成分的分类概率矩阵
        for ic = 1:size(classifications, 1) % 逐个遍历所有独立成分
            artifact_probs = classifications(ic, [2, 3, 4, 5, 6]); % 提取肌电、眼电等五类伪迹概率
            if any(artifact_probs > auto_threshold) % 任意伪迹概率超出阈值
                auto_reject_mask(ic) = true; % 将当前成分标记为待剔除伪迹
            end
        end
        fprintf('\n基于 ICLabel 自动标记了 %d 个伪迹成分 (阈值 > %.0f%%)\n',sum(auto_reject_mask), auto_threshold * 100); % 输出自动标记伪迹总数
        for ic = find(auto_reject_mask) % 遍历所有自动标记伪迹编号
            max_prob = max(classifications(ic, [2, 3, 4, 5, 6])); % 获取该成分最大伪迹概率
            [~, cat_idx] = max(classifications(ic, [2, 3, 4, 5, 6])); % 获取最大概率伪迹类别下标
            cat_names = {'Muscle','Eye','Heart','LineNoise','ChanNoise'}; % 伪迹类别名称数组
            fprintf('  IC%d: %s (%.1f%%)\n', ic, cat_names{cat_idx}, max_prob * 100); % 打印伪迹类型与概率
        end

        %% 仅保留快速审阅模式，控制台汇总IC信息，弹窗可缩放滚动
        n_ics = size(EEG.icaweights, 1); % 获取独立成分总数量
        fprintf('\n--- 快速审阅模式: IC 分类摘要 ---\n'); % 模式提示
        for ic = 1:n_ics % 遍历所有独立成分
            ic_probs = classifications(ic, :); % 读取当前成分分类概率
            [~, max_cat] = max(ic_probs); % 筛选最优分类下标
            cat_names_full = {'Brain','Muscle','Eye','Heart','LineNoise','ChanNoise','Other'}; % 全类别名称
            flag = ''; % 标记标识初始化
            if auto_reject_mask(ic) % 当前成分被自动标记伪迹
                flag = ' <<< 自动标记为伪迹'; % 添加标识文字
            end
            fprintf('  IC%02d | 脑电:%5.1f%% | 肌电:%5.1f%% | 眼动:%5.1f%% | %s%s\n',ic, ic_probs(1)*100, ic_probs(2)*100, ic_probs(3)*100,cat_names_full{max_cat}, flag); % 打印成分汇总信息
        end
        % 弹窗配置：支持缩放、滚轮滚动
        prompt = {sprintf(...
            '请输入要额外删除的 IC 编号 (多个用空格或逗号分隔)\n----------------------------------------\n自动标记的伪迹成分 (共 %d 个): %s\n----------------------------------------\n输入额外删除的 IC (留空仅用自动标记):',...
            sum(auto_reject_mask), mat2str(find(auto_reject_mask)))};
        dlgtitle = '手动确认 IC 删除列表'; % 弹窗标题
        dims = [5 80]; % 增加输入框行数，优化显示
        definput = {''}; % 默认空输入
        options.Resize = 'on'; % 开启窗口缩放，可拖拽放大滚动
        options.WindowStyle = 'modal';
        answer = inputdlg(prompt, dlgtitle, dims, definput, options); % 弹出输入窗口
        if isempty(answer) % 关闭弹窗
            manual_reject_ics = []; % 手动删除列表置空
        else
            manual_reject_ics = str2num(answer{1}); % 字符转数值
            if isempty(manual_reject_ics) % 转换失败
                manual_reject_ics = []; % 列表置空
            end
        end

        all_reject_ics = unique([find(auto_reject_mask), manual_reject_ics]); % 合并自动与手动剔除编号并去重
        if length(all_reject_ics) >= n_ics - 2 % 待剔除数量过多，防止数据损毁
            fprintf(['\n警告: 计划删除 %d/%d 个成分，将导致数据质量严重下降。仅删除前 %d 个。\n'],length(all_reject_ics), n_ics, n_ics - 3); % 超限警告
            all_reject_ics = all_reject_ics(1:max(0, n_ics - 3)); % 截断剔除列表
        end
        if isempty(all_reject_ics) % 无需要剔除的伪迹成分
            fprintf('\n>>> 没有要删除的独立成分，保留所有 IC <<<\n'); % 提示信息
        else
            fprintf('\n========================================\n'); % 分隔符
            fprintf('最终删除 %d 个独立成分: %s\n',length(all_reject_ics), mat2str(all_reject_ics)); % 输出剔除清单
            EEG = pop_subcomp(EEG, all_reject_ics, 0); % 从原始数据中移除选定伪迹独立成分
            fprintf('删除后剩余 %d 个独立成分\n', size(EEG.icaweights, 1)); % 输出剩余成分数量
            fprintf('========================================\n'); % 分隔符
        end
        EEG.setname = sprintf('%s_icaclean', EEG.setname); % 命名标记ICA伪迹剔除完成
        EEG = eeg_checkset(EEG); % 校验剔除后数据
        fprintf('伪迹去除步骤完成\n'); % 步骤结束提示
        fprintf('\n--- 执行经典频段分频段滤波 ---\n'); % 分频段滤波提示
        band_names = fieldnames(bands); % 提取五个脑电节律的字段名称
        n_bands = length(band_names); % 获取频段总个数
        EEG_bands = cell(n_bands, 1); % 新建元胞数组存储各频段滤波结果
        for b = 1:n_bands % 循环逐个处理单频段
            band_name = band_names{b}; % 获取当前频段名称
            band_range = bands.(band_name); % 获取当前频段频率上下限
            fprintf('  滤波中: %s 频段 (%.1f ~ %.1f Hz) ...\n',band_name, band_range(1), band_range(2)); % 打印当前滤波频段
            EEG_band = EEG; % 复制全频段清洗后数据用于单独滤波
            EEG_band = pop_eegfiltnew(EEG_band,'locutoff', band_range(1),'hicutoff', band_range(2),'plotfreqz', 0); % 当前频段带通滤波
            EEG_band.setname = sprintf('%s_%s', EEG.setname, band_name); % 标记频段名称
            EEG_band = eeg_checkset(EEG_band); % 校验单频段数据
            EEG_bands{b} = EEG_band; % 存入元胞数组
            fprintf('    完成: %s 频段\n', band_name); % 单频段滤波结束提示
        end
        fprintf('分频段滤波完成，共生成 %d 个频段数据集\n', n_bands); % 全频段滤波完毕提示
        fprintf('\n--- 执行脑电数据分段 (Epoch Length: %d s) ---\n', EPOCH_DURATION); % Epoch分段提示
        epoch_samples = EPOCH_DURATION * FS; % 计算单个Epoch包含的采样点数
        EEG = eeg_regepochs(EEG,'recurrence', EPOCH_DURATION,'limits', [0 EPOCH_DURATION],'rmbase', 0); % 连续脑电等长无重叠分段，关闭基线校正
        n_epochs = EEG.trials; % 获取分段后Epoch总数
        fprintf('全频段 EEG 分段完成: %d 个 epochs (每个 %d 采样点)\n', n_epochs, epoch_samples); % 输出分段结果
        EEG_bands_epoched = cell(n_bands, 1); % 元胞数组存储分段后的各频段数据
        for b = 1:n_bands % 循环处理每个频段数据
            EEG_b_epoch = EEG_bands{b}; % 读取原始分频段连续数据
            EEG_b_epoch = eeg_regepochs(EEG_b_epoch,'recurrence', EPOCH_DURATION,'limits', [0 EPOCH_DURATION],'rmbase', 0); % 同参数分段处理
            EEG_bands_epoched{b} = EEG_b_epoch; % 存入结果数组
        end
        fprintf('分频段数据分段完成\n'); % 分频段分段结束提示
        fprintf('\n--- 保存预处理结果 ---\n'); % 数据保存提示
        preprocessing_info.subject      = sub; % 记录被试编号
        preprocessing_info.condition    = condition_tags{cond}; % 记录实验任务类型
        preprocessing_info.fs           = FS; % 记录采样率参数
        preprocessing_info.n_channels   = N_CHANNELS; % 记录导联数量
        preprocessing_info.channel_labels = chan_labels; % 记录导联名称列表
        preprocessing_info.n_samples    = N_SAMPLES; % 记录原始单文件采样点数
        preprocessing_info.notch_freq   = notch_freq; % 记录陷波滤波参数
        preprocessing_info.bandpass     = [bp_low, bp_high]; % 记录带通滤波范围
        preprocessing_info.reference    = 'average'; % 记录重参考方式
        preprocessing_info.n_ica_comps  = n_ica_components; % 记录ICA成分数量
        preprocessing_info.auto_rejected_ics = find(auto_reject_mask); % 记录自动剔除IC编号
        preprocessing_info.manual_rejected_ics = manual_reject_ics; % 记录手动剔除IC编号
        preprocessing_info.bands        = bands; % 记录各频段划分参数
        preprocessing_info.epoch_length = EPOCH_DURATION; % 记录Epoch分段时长
        preprocessing_info.date         = datetime('now'); % 记录预处理执行时间
        save(out_full, 'EEG', 'EEG_bands_epoched', 'preprocessing_info', '-v7.3'); % v7.3格式保存mat文件，支持大数据
        fprintf('预处理结果已保存: %s\n', out_full); % 输出保存路径
        fprintf('文件包含:\n'); % 说明存储内容
        fprintf('  - EEG: 清洗后全频段数据 (%d epochs)\n', EEG.trials); % 全频段数据说明
        fprintf('  - EEG_bands_epoched: 5 个频段数据 (%d epochs each)\n', EEG.trials); % 分频段数据说明
        fprintf('  - preprocessing_info: 预处理参数记录\n'); % 参数信息说明
        fprintf('\n====== 被试 %02d (%s) 预处理完成 ======\n\n',sub, upper(condition_tags{cond})); % 单被试处理完毕
        clear EEG EEG_band EEG_b_epoch EEG_bands EEG_bands_epoched; % 清空单次被试大变量释放内存
    end
end
fprintf('\n========================================\n'); % 总流程结束分隔线
fprintf('全部预处理流程执行完毕！\n'); % 全部数据处理完成提示
fprintf('输出目录: %s\n', output_dir); % 输出结果文件夹路径
fprintf('========================================\n'); % 结束分隔线