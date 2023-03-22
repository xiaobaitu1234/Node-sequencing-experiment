clc;
clear;

% ʹ�õİ汾��matlab2020a
% ���ھ������ݻ���heatmap


% ���ݸ�Ŀ¼
root = fullfile('C:','Users','???','Desktop','01.�ڵ�����ʵ��');
data_source = fullfile(root,'heat_map_data'); % ��������Ŀ¼
dst_path = fullfile(root,'heat_map'); % ������Ŀ¼
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
        % ��ȡ��һ�У����б꣬��������һλ
        row = T(2:end , 1);
        row = table2array(row);
        % �ƶ�С���㣬���⾫�������ظ�
        row = row * times;
        % ��data��ɾȥ��һ��
        T = T(:,2:end);
        % ��ȡ���������б�
        col = table2array(T(1,:));
        % ��data��ɾȥ��һ��
        T = T(2:end,:);
        % tableת����
        arr = table2array(T);
        % ������ͼ
        h = heatmap(col,row,arr,'FontName','Times New Roman');
        % ѡ����ɫֵ��
        h.Colormap = jet;
        % ��ֵ����ɫ
        h.MissingDataColor = 'w';
        % ��������
        h.GridVisible = 'off';
        % ���ݼ�����
        h.Title = fname;
        
        h.CellLabelFormat = '%.2f';
        % ����X��Y����ע
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
        % �����ע
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
        
        % �����ע
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
        % ����ǩ��תΪˮƽ����
        s = struct(h);
        s.XAxis.TickLabelRotation = 0;   % horizontal
        % �洢ͼƬ
        saveas(gcf,fullfile(dst_path,[name,'.svg']));
        close
    end
end
