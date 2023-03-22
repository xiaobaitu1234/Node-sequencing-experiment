clc;
clear;

% 使用的版本是matlab2020a
% 基于矩阵数据绘制heatmap


% 数据根目录
root = fullfile('C:','Users','???','Desktop','01.节点排序实验');
data_source = fullfile(root,'heat_map_data'); % 矩阵数据目录
dst_path = fullfile(root,'heat_map'); % 结果存放目录
if ~exist(dst_path,'dir')
    mkdir(dst_path);
end

xlabel = 'ait_shell_level';
times = 100000;

names = {'Actors','AstroPh','CondMat','GrQc','HepPh','HepTh','Jazz','Musae-FR','NetScience','Wiki-Vote'};
names = names';

ylabels = {'betweenesscentrality','degree','k_shell_level'};
ylabels = ylabels';

for j = 1:size(names)
    fname = cell2mat(names(j));
    for i = 1:size(ylabels)
        ylabel = cell2mat(ylabels(i));
        name = [fname,'-row-',xlabel,'-col-',ylabel,'-result'];
        filename = fullfile(data_source,[name,'.csv']);
        T = readtable(filename);
        % 获取第一列，即行标，但舍弃第一位
        row = T(2:end , 1);
        row = table2array(row);
        % 移动小数点，避免精度问题重复
        row = row * times;
        % 从data中删去第一列
        T = T(:,2:end);
        % 获取列名，即列标
        col = table2array(T(1,:));
        % 从data中删去第一行
        T = T(2:end,:);
        % table转数组
        arr = table2array(T);
        % 绘制热图
        h = heatmap(col,row,arr,'FontName','Times New Roman');
        % 选定颜色值域
        h.Colormap = jet;
        % 空值填充白色
        h.MissingDataColor = 'w';
        % 隐藏网格
        h.GridVisible = 'off';
        % 数据集名称
        h.Title = fname;
        
        h.CellLabelFormat = '%.2f';
        % 绘制X，Y轴题注
        if size(xlabel) == size('ait_shell_level')
            h.XLabel = 'IKs';
        else
            h.XLabel = xlabel;
        end

        if size(ylabel) == size('degree')
            h.YLabel = 'DC';
        else if size(ylabel) == size('betweenesscentrality')
                h.YLabel = 'BC';
            else if size(ylabel) == size('k_shell_level')
                    h.YLabel = 'Ks';ylabel;
                end
            end
        end

        ax = gca;
        row_tick_nums = 5;
        col_tick_nums = 5;
        % 纵轴标注
        h.YDisplayData = row;
        block = ' ';
        row_min = min(row)/times;
        row_max = max(row)/times;
        row_num = size(row,1);
        row_tick_step = fix(row_num/row_tick_nums);
        if size(ylabel) == size('degree')
            row_label_step = fix((row_max-row_min)/(row_tick_nums-1));
        else if size(ylabel) == size('betweenesscentrality')
                row_label_step = ((row_max-row_min)/(row_tick_nums-1));
            else if size(ylabel) == size('k_shell_level')
                    row_label_step = fix((row_max-row_min)/(row_tick_nums-1));
                end
            end
        end
        rows = {};
        for step = 1:row_num
            rows(end+1) = cellstr(block);
        end
        for step = 0:row_tick_nums-1
            temp = step*row_tick_step+1;
            str = sprintf('%.2E',row_max-row_label_step*step);
            rows(temp) = cellstr(str);
        end
        
        rows(end) = cellstr(sprintf('%.0E',0));
        h.YDisplayLabels = rows;
        
        % 横轴标注
        h.XDisplayData = col;
        block = ' ';
        col_min = min(col);
        col_max = max(col);
        col_num = size(col',1);
        col_tick_step = fix(col_num/col_tick_nums);
        col_label_step = fix((col_max-col_min)/(col_tick_nums-1));
        cols = {};
        for step = 1:col_num
            cols(end+1) = cellstr(block);
        end
        for step = 0:col_tick_nums-1
            temp = col_num - step*col_tick_step;
            str = sprintf('%.2E',col_max-col_label_step*step);
            cols(temp) = cellstr(str);
        end
        cols(1) = cellstr(sprintf('%.0E',0));
        h.XDisplayLabels = cols;
        % 将标签旋转为水平方向
        s = struct(h);
        s.XAxis.TickLabelRotation = 0;   % horizontal
        % 存储图片
        saveas(gcf,fullfile(dst_path,[name,'.svg']));
        close
    end
end
