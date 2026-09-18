%读取预处理后的.mat文件 -> 匹配指定脑电导联 -> 对所有试次(epoch)做叠加平均 -> 绘制时域波形 -> 批量保存图片
clear; clc; close all;
% ===== 1. 定义数据与图片保存路径 =====
% 预处理后EEG数据所在文件夹路径
data_dir = fullfile(fileparts(mfilename('fullpath')), 'preprocessed_data');
% 在数据目录下新建visualization子文件夹用于存图
fig_dir  = fullfile(data_dir,'visualization');
if ~exist(fig_dir,'dir'), mkdir(fig_dir); end
% ===== 2. 指定需要可视化的脑电导联 =====
plot_channels = {'Fp1','F3','C3','P3','O1'};
% ===== 3. 遍历文件夹中所有预处理后的MAT数据文件 =====
% 读取目标文件夹下所有后缀为 _preprocessed.mat 的文件，返回文件结构体数组
files = dir(fullfile(data_dir,'*_preprocessed.mat'));
% 逐个循环处理每一个EEG数据文件
for k = 1:length(files)
    % 在命令行打印当前正在处理的文件名，方便查看运行进度
    fprintf('处理 %s\n',files(k).name);
    % 加载当前mat文件中的两个核心变量：EEG、preprocessing_info
    load(fullfile(files(k).folder, files(k).name),'EEG','preprocessing_info');
    % 初始化数组，用于存储待绘制导联对应的索引编号
    channels = [];
    % 遍历需要绘制的所有导联，匹配导联标签、查找对应索引
    for i = 1:length(plot_channels)
        % 比对导联名称：在全部导联标签中找到当前目标导联的位置索引
        idx = find(strcmp(preprocessing_info.channel_labels, plot_channels{i}));
        % 若成功找到该导联（索引非空），将索引存入数组
        if ~isempty(idx), channels(end+1) = idx; end
    end
    % EEG.data 标准维度说明：[通道数 × 采样点数 × 试次(epochs)数]
    % mean(...,3)：在第3维（试次维度）求均值，实现所有试次叠加平均，消除单次试次随机噪声
    avg_signal = mean(EEG.data,3);  
    % 生成时间轴：采样点序号 / 采样率(srate)，将采样点转换为 秒(s) 单位
    t = (0:size(avg_signal,2)-1)/EEG.srate;  
    % 创建图像窗口：visible='off' 后台绘图，设置窗口尺寸[左,上,宽,高]
    figure('visible','off','Position',[100 100 1200 800]);
    % 循环绘制每个选定导联的平均脑电波形
    for i = 1:length(channels)
        % 搭建子图布局：总行数=导联数，单列，当前第i个子图
        subplot(length(channels),1,i)
        % 绘制时域波形：横坐标时间t，纵坐标该导联的平均脑电信号，线宽设为1
        plot(t, avg_signal(channels(i),:),'LineWidth',1)
        % 设置Y轴标签：当前导联名称
        ylabel(plot_channels{i})
        % 显示网格线，方便读取波形幅值与时间
        grid on
    end
    % 设置全局X轴标签（所有子图共用）：时间(秒)
    xlabel('Time (s)')
    % 设置整张图的总标题
    sgtitle(sprintf('%s : 平均 EEG 波形', erase(files(k).name,'.mat')))
    % 将绘制好的图像保存为PNG格式
    saveas(gcf, fullfile(fig_dir, [erase(files(k).name,'.mat') '_AvgEEG.png']));
    % 关闭当前图像窗口，释放内存，避免批量绘图造成卡顿
    close(gcf);
end
disp('全部图像生成完成！');