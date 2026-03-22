clear, clc, close all

% load D2.mat
%
% [pks1,locs1] = findpeaks(d, 'MinPeakProminence', 1.5);
%
% figure(1)
% plot(d)
% hold on
% scatter(Index, d(Index))
% scatter(locs1,pks1)
% hold off

load D3.mat

fs = 25000;
% df = bandpass(d,[450 2050],fs);
df = bandpass(d,[300 3000],fs);

%% Extract features
disp('*** Extracting features...');
n = length(df);
dt = 1/fs;
% Convert window length from seconds to samples
row = 10;
% Convert overlap from % to samples
overlap = 1;

% Work out how many windows we will have
% Effective window length...
len = row - overlap;
nwin = floor((n - overlap) / (len))-1;
% Storage
mav = zeros(nwin, 1);
for j = 1:nwin
    idx = j*len; % Starting position
    win = df(idx:idx+row);

    % Extract features
    mav(j) = mean(abs(win));
end
clear len idx win j
% Revise dt based on windowing
dt = dt*(row - overlap);
n = length(mav);
t = (0:n-1)*dt;
flp = 0.1;

% Smooth features
[b, a] = butter(2, 0.05, 'low');
df2 = filtfilt(b, a, df);

ts = 3*(median(abs(df-mean(df))/0.7645));
[pks1,locs1] = findpeaks(df, 'MinPeakProminence', 1.5, 'MinPeakHeight', ts);
Index = zeros(1, length(pks1));

for i=1:length(pks1)
    Index(i) = locs1(i) - 10;
end

figure(2)
plot(mav)
% hold on
% scatter(Index, d(Index))
% scatter(locs1,pks1)
% hold off

% sigma = median(mean(d)/0.6745);
% ts = sigma*5;
% spike = zeros(1, length(d)-50);
% new_index = zeros(1,length(Index));
%
% j = 1;
% for i=26:length(d)-25
%     if d(i) > ts
%         spike(j) = d(i);
%     else
%         spike(j) = ts;
%     end
%     j = j + 1;
% end
%
% [pks,locs] = findpeaks(spike, 'MinPeakProminence', 0.5);
%
% figure(2)
% plot(spike)
% hold on
% scatter(Index, d(Index))
% scatter(locs,pks)
% hold off


