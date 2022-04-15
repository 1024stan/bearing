clc,clear;


bearing_name = './data/Bearing3_2';

mk_data_filename = [bearing_name '_M-K检测.csv'];
mk_data = csvread(mk_data_filename,1,1);


 
% %% 原始数据与预测数据对比
% old_x = old_data(:, 1);
% now_y1 = predict_data(:, 1);
% now_y2 = sum(predict_data(:,2:end-1), 2);
% figure 
% hold on
% plot(old_x);
% plot(now_y1);
% plot(now_y2);
% hold
% legend('原始数据','单维预测','多维预测')




%% 画突变点
for i =1:5
    bearing_name = ['./data/Bearing3_' int2str(i)];
    mk_data_filename = [bearing_name '_M-K检测.csv'];
    mk_data = csvread(mk_data_filename, 1, 1);
    times = size(mk_data, 1);
    time1 = 1:times;
    
    subplot(3,2,i)
    plot(time1, mk_data(:, 1),'k','LineWidth', 1);
    hold on
    target = mk_data(:,end);
    t1 = find(target==1, 1);
    t2 = find(target==2, 1);
    plot(t1, mk_data(t1, 1), 'y.', 'MarkerSize', 18);
    plot(t2, mk_data(t2, 1), 'r.', 'MarkerSize', 18);
    if i ==5
        legend('原始数据','异常工作突变点','无法工作突变点');
    end
    xlabel('时间');
    ylabel('加速度');
    hold off
    set(gca,'FontSize',20);
end




%% 画预测数据和原始数据的对比图
% 
% figure
% hold on
% for i=2:size(predict_data, 2)
%     plot(time1, old_data(:, i), 'r','LineWidth',1.5);
%     plot(time2, predict_data(:, i), 'b','LineWidth',1.5);
% end
% xlabel('时间');
% ylabel('IMF特征');
% hold off
% legend('原始数据','预测数据');

%% 画预测数据与实际数据的对比图
file_name = '训练参数和误差记录.xlsx';
for i =1:5
    bearing_name = ['Bearing3_' int2str(i)];
    % mk_data_filename = [bearing_name '_M-K检测.csv'];
    svm_class_data = xlsread(file_name, bearing_name, );
    times = size(mk_data, 1);
    time1 = 1:times;
    
    subplot(3,2,i)
    plot(time1, mk_data(:, 1),'k','LineWidth', 1);
    hold on
    target = mk_data(:,end);
    t1 = find(target==1, 1);
    t2 = find(target==2, 1);
    plot(t1, mk_data(t1, 1), 'y.', 'MarkerSize', 18);
    plot(t2, mk_data(t2, 1), 'r.', 'MarkerSize', 18);
    if i ==5
        legend('原始数据','异常工作突变点','无法工作突变点');
    end
    xlabel('时间');
    ylabel('加速度');
    hold off
    set(gca,'FontSize',20);
end



