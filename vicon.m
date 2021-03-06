% M = csvread('speer_ssh_slow.csv', 6,1 );
% 
% subset = M(2845:2988,:)
% rx = subset(:,2)
% ry = subset(:,3)
% rz = subset(:,4)

M = csvread('speer_ssh.csv', 6,1 );
%zeros = find(M(:,2)==0);
%M(zeros,:) = [];


% spin starts at frame 420
% data starts at 3817

% vicon_start_in_s = (4150 - 3817) / 250;
% vicon_end_in_s = (4175 - 3817) / 250;
% vicon_start_in_frames = vicon_start_in_s * 90
% vicon_end_in_frames = vicon_end_in_s * 90
% 
% subset = M(4150 - 1:4175 - 1,:);
% 
% %subset = M(4222-1:4246-1,:)
% rx = median(diff(subset(:,2))) * 250;
% ry = median(diff(subset(:,3))) * 250;
% rz = median(diff(subset(:,4))) * 250;
% norm([ry rz])

v_offset = 3817;
f_offset = 420;

beg_f = 1100;
end_f = 1200;

f = 1100;

t_start = (beg_f - f_offset)  / 90;
beg_v = t_start * 250 + v_offset

t_end = (end_f - f_offset)  / 90;
end_v = t_end * 250 + v_offset

subset = M(round(beg_v-1):round(end_v-1),:);
zeros = find(subset(:,2)==0);
subset(zeros,:) = [];

rx = median(diff(subset(:,2))) * 250;
ry = median(diff(subset(:,3))) * 250;
rz = median(diff(subset(:,4))) * 250;

mag = norm([rz ry]);

%% empty
M = csvread('/home/smorad/spin_odom/speer_data/speer_2_6_empty.csv', 6,1 );
v_offset = 1245;
f_offset = 700;

beg_f = 1100;
end_f = beg_f + 200;

t_start = (beg_f - f_offset)  / 90;
beg_v = t_start * 100 + v_offset

t_end = (end_f - f_offset)  / 90;
end_v = t_end * 100 + v_offset

subset = M(round(beg_v-1):round(end_v-1),:);
%zeros = find(subset(:,2)==0);
%subset(zeros,:) = [];
% 
% rx = median(diff(subset(:,2))) * 100;
% ry = median(diff(subset(:,3))) * 100;
% rz = median(diff(subset(:,4))) * 100;

rx = median(diff(subset(:,2))) * 100;
ry = median(diff(subset(:,3))) * 100;
rz = median(diff(subset(:,4))) * 100;


mag = norm([rz ry])
slope = tan(ry/ rz)

%% chairs
M = csvread('/home/smorad/spin_odom/speer_data/speer_2_5_chairs.csv', 6,1 );
v_offset = 1275;
f_offset = 675;

beg_f = 800;
end_f = 900;

t_start = (beg_f - f_offset)  / 90;
beg_v = t_start * 100 + v_offset

t_end = (end_f - f_offset)  / 90;
end_v = t_end * 100 + v_offset

subset = M(round(beg_v-1):round(end_v-1),:);
%zeros = find(subset(:,2)<=0);
%subset(zeros,:) = [];
% 
% rx = median(diff(subset(:,2))) * 100;
% ry = median(diff(subset(:,3))) * 100;
% rz = median(diff(subset(:,4))) * 100;

rx = median(diff(subset(:,2))) * 100;
ry = (0.1899 - 0.02853) / (1310 - 1286) *100;%median(diff(subset(:,3))) * 100;
rz = (0.2021 - 0.05171)/ (1384-1369) * 100;%median(diff(subset(:,4))) * 100;

% fitFn = @(subset) polyfit(1:length(subset(:,2)), subset(:,2)', 1);
% evalFn = @(model, subset) sum((subset(:,2) - polyval(model, 1:length(subset(:,2)))) .^2, 2); 
% [model, inliers] = ransac(subset(:,2), fitFn, evalFn, 2, 


mag = norm([rz ry])
slope = tan(ry/ rz)

%% pillars
M = csvread('/home/smorad/spin_odom/speer_data/speer_2_4.csv', 6,1 );
v_offset = 1378;
f_offset = 570;

beg_f = 970;
end_f = beg_f + 45;

t_start = (beg_f - f_offset)  / 90;
beg_v = t_start * 100 + v_offset

t_end = (end_f - f_offset)  / 90;
end_v = t_end * 100 + v_offset

subset = M(round(beg_v-1):round(end_v-1),:);
%zeros = find(subset(:,2)==0);
%subset(zeros,:) = [];
% 
% rx = median(diff(subset(:,2))) * 100;
% ry = median(diff(subset(:,3))) * 100;
% rz = median(diff(subset(:,4))) * 100;

rx = median(diff(subset(:,2))) * 100;
ry = median(diff(subset(:,3))) * 100;%(0.2021 - 0.05171)/ (1384-1369) * 100;
rz = median(diff(subset(:,4))) * 100;%(5.78 - 0.6778) / (1413 - 1368) * 100;%


mag = norm([rz ry])
slope = tan(ry/ rz)