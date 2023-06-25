close all
clear all
bag1=rosbag("imuJ.bag");
bag2=rosbag("magJ.bag");
bsel1=select(bag1,"Topic","imu");
bsel2=select(bag2,"Topic","mag");
msgStruct1=readMessages(bsel1,'DataFormat','struct');
msgStruct2=readMessages(bsel2,'DataFormat','struct');
or_x=cellfun(@(m) double(m.Orientation.X),msgStruct1);
or_y=cellfun(@(m) double(m.Orientation.Y),msgStruct1);
or_z=cellfun(@(m) double(m.Orientation.Z),msgStruct1);
or_w=cellfun(@(m) double(m.Orientation.W),msgStruct1);
angv_x=cellfun(@(m) double(m.AngularVelocity.X),msgStruct1);
angv_y=cellfun(@(m) double(m.AngularVelocity.Y),msgStruct1);
angv_z=cellfun(@(m) double(m.AngularVelocity.Z),msgStruct1);
linacc_x=cellfun(@(m) double(m.LinearAcceleration.X),msgStruct1);
linacc_y=cellfun(@(m) double(m.LinearAcceleration.Y),msgStruct1);
linacc_z=cellfun(@(m) double(m.LinearAcceleration.Z),msgStruct1);
mag_x=cellfun(@(m) double(m.MagneticField_.X),msgStruct2);
mag_y=cellfun(@(m) double(m.MagneticField_.Y),msgStruct2);
mag_z=cellfun(@(m) double(m.MagneticField_.Z),msgStruct2);

%% 15 minute data analysis
mean_accelx=mean(linacc_x);
mean_accely=mean(linacc_y);
mean_accelz=mean(linacc_z);
mean_angvx=mean(angv_x);
mean_angvy=mean(angv_y);
mean_angvz=mean(angv_z);
mean_magx=mean(mag_x);
mean_magy=mean(mag_y);
mean_magz=mean(mag_z);
std_accelx=std(linacc_x);
std_accely=std(linacc_y);
std_accelz=std(linacc_z);
std_angvx=std(angv_x);
std_angvy=std(angv_y);
std_angvz=std(angv_z);
std_magx=std(mag_x);
std_magy=std(mag_y);
std_magz=std(mag_z);
err_accelx=abs(linacc_x-mean_accelx);
err_accely=abs(linacc_y-mean_accely);
err_accelz=abs(linacc_z-mean_accelz);
err_angvx=abs(angv_x-mean_angvx);
err_angvy=abs(angv_y-mean_angvy);
err_angvz=abs(angv_z-mean_angvz);
err_magx=abs(mag_x-mean_magx);
err_magy=abs(mag_y-mean_magy);
err_magz=abs(mag_z-mean_magz);
subplot(2,2,1)
plot(or_x)
title("Quartenion X Data")
ylabel("Quartenion X values")
xlabel("Time(seconds)")
subplot(2,2,2)
plot(or_y)
title("Quartenion Y Data")
ylabel("Quartenion Y values")
xlabel("Time(seconds)")
subplot(2,2,3)
plot(or_z)
title("Quartenion Z Data")
ylabel("Quartenion Z values")
xlabel("Time(seconds)")
subplot(2,2,4)
plot(or_w)
title("Quartenion W Data")
ylabel("Quartenion W values")
xlabel("Time(seconds)")

figure
subplot(3,1,1)
plot(linacc_x,'b+')
title(" Linear acceleration in X-axis Data")
ylabel("Linear acceleration(m/s^2)")    
xlabel("Time(seconds)")
subplot(3,1,2)
plot(linacc_y,'b+')
title(" Linear acceleration in Y-axis Data")
ylabel("Linear acceleration(m/s^2)")
xlabel("Time(seconds)")
subplot(3,1,3)
plot(linacc_z,'b+')
title(" Linear acceleration in Z-axis Data")
ylabel("Linear acceleration(m/s^2)")
xlabel("Time(seconds)")

figure
subplot(3,1,1)
plot(angv_x,'b+')
title(" Angular Velocity in X-axis Data")
ylabel("Angular Velocity(rad/s)")
xlabel("Time(seconds)")
subplot(3,1,2)
plot(angv_y,'b+')
title(" Angular Velocity in Y-axis Data")
ylabel("Angular Velocity(rad/s)")
xlabel("Time(seconds)")
subplot(3,1,3)
plot(angv_z,'b+')
title(" Angular Velocity in Z-axis Data")
ylabel("Angular Velocity(rad/s)")
xlabel("Time(seconds)")

figure
subplot(3,1,1)
plot(mag_x,'b+')
title(" Magnetometer measurment in X-axis Data")
ylabel("Magnetometer measurment (Gauss)")
xlabel("Time(seconds)")
subplot(3,1,2)
plot(mag_y,'b+')
title(" Magnetometer measurment in Y-axis Data")
ylabel("Magnetometer measurment (Gauss)")
xlabel("Time(seconds)")
subplot(3,1,3)
plot(mag_z,'b+')
title(" Magnetometer measurment in Z-axis Data")
ylabel("Magnetometer measurment (Gauss)")
xlabel("Time(seconds)")

figure
subplot(3,3,1)
plot(err_accelx,'b+')
title(" Linear acceleration in X-axis absolute error about mean data")
ylabel("Absolute value of error(m/s^2)")
xlabel("Time(seconds)")
subplot(3,3,2)
plot(err_accely,'b+')
title(" Linear acceleration in Y-axis absolute error about mean data")
ylabel("Absolute value of error(m/s^2)")
xlabel("Time(seconds)")
subplot(3,3,3)
plot(err_accelz,'b+')
title(" Linear acceleration in Z-axis absolute error about mean data")
ylabel("Absolute value of error(m/s^2)")
xlabel("Time(seconds)")
subplot(3,3,4)
plot(err_angvx,'b+')
title(" Angular velocity in X-axis absolute error about mean data")
ylabel("Absolute value of error(rad/s)")
xlabel("Time(seconds)")
subplot(3,3,5)
plot(err_angvy,'b+')
title(" Angular velocity in Y-axis absolute error about mean data")
ylabel("Absolute value of error(rad/s)")
xlabel("Time(seconds)")
subplot(3,3,6)
plot(err_angvz,'b+')
title(" Angular velocity in Z-axis absolute error about mean data")
ylabel("Absolute value of error(rad/s)")
xlabel("Time(seconds)")
subplot(3,3,7)
plot(err_magx,'b+')
title(" Magnetometer readings in X-axis absolute error about mean data")
ylabel("Absolute value of error(Gauss)")
xlabel("Time(seconds)")
subplot(3,3,8)
plot(err_magy,'b+')
title(" Magnetometer readings in Y-axis absolute error about mean data")
ylabel("Absolute value of error(Gauss)")
xlabel("Time(seconds)")
subplot(3,3,9)
plot(err_magz,'b+')
title(" Magnetometer readings in Z-axis absolute error about mean data")
ylabel("Absolute value of error(Gauss)")
xlabel("Time(seconds)")

figure
subplot(3,3,1)
normd_accelx=normpdf(linacc_x,mean_accelx,std_accelx);
plot(linacc_x,normd_accelx,'b+')
title(" Normal distribution of linear acceleration in X-axis")
ylabel("distribution of points")
xlabel("Linear acceleration values(m/s^2)")
subplot(3,3,2)
normd_accely=normpdf(linacc_y,mean_accely,std_accely);
plot(linacc_y,normd_accely,'b+')
title(" Normal distribution of linear acceleration in Y-axis")
ylabel("distribution of points")
xlabel("Linear acceleration values(m/s^2)")
subplot(3,3,3)
normd_accelz=normpdf(linacc_z,mean_accelz,std_accelz);
plot(linacc_z,normd_accelz,'b+')
title(" Normal distribution of linear acceleration in X-axis")
ylabel("distribution of points")
xlabel("Linear acceleration values(m/s^2)")
subplot(3,3,4)
normd_angvx=normpdf(angv_x,mean_angvx,std_angvx);
plot(angv_x,normd_angvx,'b+')
title(" Normal distribution of angular velocity in X-axis")
ylabel("distribution of points")
xlabel("Angular velocity values(rad/s)")
subplot(3,3,5)
normd_angvy=normpdf(angv_y,mean_angvy,std_angvy);
plot(angv_y,normd_angvy,'b+')
title(" Normal distribution of angular velocity in Y-axis")
ylabel("distribution of points")
xlabel("Angular velocity values(rad/s)")
subplot(3,3,6)
normd_angvz=normpdf(angv_z,mean_angvz,std_angvz);
plot(angv_z,normd_angvz,'b+')
title(" Normal distribution of angular velocity in Z-axis")
ylabel("distribution of points")
xlabel("Angular velocity values(rad/s)")
subplot(3,3,7)
normd_magx=normpdf(mag_x,mean_magx,std_magx);
plot(mag_x,normd_magx,'b+')
title(" Normal distribution of magnetometer measurements in X-axis")
ylabel("distribution of points")
xlabel("Magnetometer measuremnet values(Gauss)")
subplot(3,3,8)
normd_magy=normpdf(mag_y,mean_magy,std_magy);
plot(mag_y,normd_magy,'b+')
title(" Normal distribution of magnetometer measurements in Y-axis")
ylabel("distribution of points")
xlabel("Magnetometer measuremnet values(Gauss)")
subplot(3,3,9)
normd_magz=normpdf(mag_z,mean_magz,std_magz);
plot(mag_z,normd_magz,'b+')
title(" Normal distribution of magnetometer measurements in Z-axis")
ylabel("distribution of points")
xlabel("Magnetometer measuremnet values(Gauss)")
    
%%
% 5 hour Data analysis
bag3=rosbag("imu_5hr.bag");
bsel3=select(bag1,"Topic","imu");
msgStruct3=readMessages(bsel1,'DataFormat','struct');
acl_x= cellfun(@(m) double(m.LinearAcceleration.X),msgStruct3);
acl_y= cellfun(@(m) double(m.LinearAcceleration.Y),msgStruct3);
acl_z= cellfun(@(m) double(m.LinearAcceleration.Z),msgStruct3);
angv_x2= cellfun(@(m) double(m.AngularVelocity.X),msgStruct3);
angv_y2= cellfun(@(m) double(m.AngularVelocity.Y),msgStruct3);
angv_z2= cellfun(@(m) double(m.AngularVelocity.Z),msgStruct3);
Fs= 40;
t0 = 1/Fs;

theta = cumsum(acl_x, 1)*t0;
maxNumM = 100;
L = size(theta, 1);
maxM = 2.^floor(log2(L/2));
m = logspace(log10(1), log10(maxM), maxNumM).';
m = ceil(m); % m must be an integer.
m = unique(m); % Remove duplicates.

tau = m*t0;

avar = zeros(numel(m), 1);
for i = 1:numel(m)
    mi = m(i);
    avar(i,:) = sum( ...
        (theta(1+2*mi:L) - 2*theta(1+mi:L-mi) + theta(1:L-2*mi)).^2, 1);
end
avar = avar ./ (2*tau.^2 .* (L - 2*m));
adev = sqrt(avar);
% Find the index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = -0.5;
logtau = log10(tau);
logadev = log10(adev);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));

% Find the y-intercept of the line.
b = logadev(i) - slope*logtau(i);

% Determine the angle random walk coefficient from the line.
logN = slope*log(1) + b;
N = 10^logN

% Plot the results.
tauN = 1;
lineN = N ./ sqrt(tau);

% Find the index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = 0.5;
logtau = log10(tau);
logadev = log10(adev);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));

% Find the y-intercept of the line.
b = logadev(i) - slope*logtau(i);

% Determine the rate random walk coefficient from the line.
logK = slope*log10(3) + b;
K = 10^logK

% Plot the results.
tauK = 3;
lineK = K .* sqrt(tau/3);

% Find the index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = 0;
logtau = log10(tau);
logadev = log10(adev);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));

% Find the y-intercept of the line.
b = logadev(i) - slope*logtau(i);

% Determine the bias instability coefficient from the line.
scfB = sqrt(2*log(2)/pi);
logB = b - log10(scfB);
B = 10^logB

% Plot the results.
tauB = tau(i);
lineB = B * scfB * ones(size(tau));

tauParams = [tauN, tauK, tauB];
params = [N, K, scfB*B];
figure
loglog(tau, adev, tau, [lineN, lineK, lineB-0.0319], '--', ...
    tauParams, params, 'o')
title('Allan Deviation of Linear Acceleration about X-axis with Noise Parameters')
xlabel('\tau')
ylabel('\sigma(\tau)')
legend('$\sigma (m/s)$', '$\sigma_N ((m/s)/\sqrt{Hz})$', ...
    '$\sigma_K ((m/s)\sqrt{Hz})$', '$\sigma_B (m/s)$', 'Interpreter', 'latex')
text(tauParams, params, {'N', 'K', '0.664B'})
grid on
axis equal
% Generating a simulated curve
generateSimulatedData = false;

if generateSimulatedData
    % Set the gyroscope parameters to the noise parameters determined
    % above.
    gyro = gyroparams('NoiseDensity', N, 'RandomWalk', K, ...
        'BiasInstability', B);
    omegaSim = helperAllanVarianceExample(L, Fs, gyro);
else
    load('SimulatedSingleAxisGyroscope', 'omegaSim')
end
[avarSim, tauSim] = allanvar(omegaSim, 'octave', Fs);
adevSim = sqrt(avarSim);
adevSim = mean(adevSim, 2); % Use the mean of the simulations.

figure
loglog(tau, adev, tauSim, adevSim, '--')
title('Allan Deviation of HW and Simulation of Linear acceleration about X-axis')
xlabel('\tau');
ylabel('\sigma(\tau)')
legend('HW', 'SIM')
grid on
axis equal

theta = cumsum(acl_y, 1)*t0;
maxNumM = 100;
L = size(theta, 1);
maxM = 2.^floor(log2(L/2));
m = logspace(log10(1), log10(maxM), maxNumM).';
m = ceil(m); % m must be an integer.
m = unique(m); % Remove duplicates.

tau = m*t0;

avar = zeros(numel(m), 1);
for i = 1:numel(m)
    mi = m(i);
    avar(i,:) = sum( ...
        (theta(1+2*mi:L) - 2*theta(1+mi:L-mi) + theta(1:L-2*mi)).^2, 1);
end
avar = avar ./ (2*tau.^2 .* (L - 2*m));
adev = sqrt(avar);
% Find the index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = -0.5;
logtau = log10(tau);
logadev = log10(adev);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));

% Find the y-intercept of the line.
b = logadev(i) - slope*logtau(i);

% Determine the angle random walk coefficient from the line.
logN = slope*log(1) + b;
N = 10^logN

% Plot the results.
tauN = 1;
lineN = N ./ sqrt(tau);

% Find the index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = 0.5;
logtau = log10(tau);
logadev = log10(adev);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));

% Find the y-intercept of the line.
b = logadev(i) - slope*logtau(i);

% Determine the rate random walk coefficient from the line.
logK = slope*log10(3) + b;
K = 10^logK

% Plot the results.
tauK = 3;
lineK = K .* sqrt(tau/3);

% Find the index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = 0;
logtau = log10(tau);
logadev = log10(adev);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));

% Find the y-intercept of the line.
b = logadev(i) - slope*logtau(i);

% Determine the bias instability coefficient from the line.
scfB = sqrt(2*log(2)/pi);
logB = b - log10(scfB);
B = 10^logB

% Plot the results.
tauB = tau(i);
lineB = B * scfB * ones(size(tau));

tauParams = [tauN, tauK, tauB];
params = [N, K, scfB*B];
figure
loglog(tau, adev, tau, [lineN, lineK, lineB], '--', ...
    tauParams, params, 'o')
title('Allan Deviation of Linear Acceleration about Y-axis with Noise Parameters')
xlabel('\tau')
ylabel('\sigma(\tau)')
legend('$\sigma (m/s)$', '$\sigma_N ((m/s)/\sqrt{Hz})$', ...
    '$\sigma_K ((m/s)\sqrt{Hz})$', '$\sigma_B (m/s)$', 'Interpreter', 'latex')
text(tauParams, params, {'N', 'K', '0.664B'})
grid on
axis equal
% Generating a simulated curve
generateSimulatedData = false;

if generateSimulatedData
    % Set the gyroscope parameters to the noise parameters determined
    % above.
    gyro = gyroparams('NoiseDensity', N, 'RandomWalk', K, ...
        'BiasInstability', B);
    omegaSim = helperAllanVarianceExample(L, Fs, gyro);
else
    load('SimulatedSingleAxisGyroscope', 'omegaSim')
end
[avarSim, tauSim] = allanvar(omegaSim, 'octave', Fs);
adevSim = sqrt(avarSim);
adevSim = mean(adevSim, 2); % Use the mean of the simulations.

figure
loglog(tau, adev, tauSim, adevSim, '--')
title('Allan Deviation of HW and Simulation of Linear acceleration about Y-axis')
xlabel('\tau');
ylabel('\sigma(\tau)')
legend('HW', 'SIM')
grid on
axis equal

theta = cumsum(acl_z, 1)*t0;
maxNumM = 100;
L = size(theta, 1);
maxM = 2.^floor(log2(L/2));
m = logspace(log10(1), log10(maxM), maxNumM).';
m = ceil(m); % m must be an integer.
m = unique(m); % Remove duplicates.

tau = m*t0;

avar = zeros(numel(m), 1);
for i = 1:numel(m)
    mi = m(i);
    avar(i,:) = sum( ...
        (theta(1+2*mi:L) - 2*theta(1+mi:L-mi) + theta(1:L-2*mi)).^2, 1);
end
avar = avar ./ (2*tau.^2 .* (L - 2*m));
adev = sqrt(avar);
% Find the index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = -0.5;
logtau = log10(tau);
logadev = log10(adev);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));

% Find the y-intercept of the line.
b = logadev(i) - slope*logtau(i);

% Determine the angle random walk coefficient from the line.
logN = slope*log(1) + b;
N = 10^logN

% Plot the results.
tauN = 1;
lineN = N ./ sqrt(tau);

% Find the index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = 0.5;
logtau = log10(tau);
logadev = log10(adev);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));

% Find the y-intercept of the line.
b = logadev(i) - slope*logtau(i);

% Determine the rate random walk coefficient from the line.
logK = slope*log10(3) + b;
K = 10^logK

% Plot the results.
tauK = 3;
lineK = K .* sqrt(tau/3);

% Find the index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = 0;
logtau = log10(tau);
logadev = log10(adev);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));

% Find the y-intercept of the line.
b = logadev(i) - slope*logtau(i);

% Determine the bias instability coefficient from the line.
scfB = sqrt(2*log(2)/pi);
logB = b - log10(scfB);
B = 10^logB

% Plot the results.
tauB = tau(i);
lineB = B * scfB * ones(size(tau));

tauParams = [tauN, tauK, tauB];
params = [N, K, scfB*B];
figure
loglog(tau, adev, tau, [lineN, lineK, lineB], '--', ...
    tauParams, params, 'o')
title('Allan Deviation of Linear Acceleration about Z-axis with Noise Parameters')
xlabel('\tau')
ylabel('\sigma(\tau)')
legend('$\sigma (m/s)$', '$\sigma_N ((m/s)/\sqrt{Hz})$', ...
    '$\sigma_K ((m/s)\sqrt{Hz})$', '$\sigma_B (m/s)$', 'Interpreter', 'latex')
text(tauParams, params, {'N', 'K', '0.664B'})
grid on
axis equal
% Generating a simulated curve
generateSimulatedData = false;

if generateSimulatedData
    % Set the gyroscope parameters to the noise parameters determined
    % above.
    gyro = gyroparams('NoiseDensity', N, 'RandomWalk', K, ...
        'BiasInstability', B);
    omegaSim = helperAllanVarianceExample(L, Fs, gyro);
else
    load('SimulatedSingleAxisGyroscope', 'omegaSim')
end
[avarSim, tauSim] = allanvar(omegaSim, 'octave', Fs);
adevSim = sqrt(avarSim);
adevSim = mean(adevSim, 2); % Use the mean of the simulations.

figure
loglog(tau, adev, tauSim, adevSim, '--')
title('Allan Deviation of HW and Simulation of Linear acceleration about Z-axis')
xlabel('\tau');
ylabel('\sigma(\tau)')
legend('HW', 'SIM')
grid on
axis equal

theta = cumsum(angv_x2, 1)*t0;
maxNumM = 100;
L = size(theta, 1);
maxM = 2.^floor(log2(L/2));
m = logspace(log10(1), log10(maxM), maxNumM).';
m = ceil(m); % m must be an integer.
m = unique(m); % Remove duplicates.

tau = m*t0;

avar = zeros(numel(m), 1);
for i = 1:numel(m)
    mi = m(i);
    avar(i,:) = sum( ...
        (theta(1+2*mi:L) - 2*theta(1+mi:L-mi) + theta(1:L-2*mi)).^2, 1);
end
avar = avar ./ (2*tau.^2 .* (L - 2*m));
adev = sqrt(avar);
% Find the index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = -0.5;
logtau = log10(tau);
logadev = log10(adev);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));

% Find the y-intercept of the line.
b = logadev(i) - slope*logtau(i);

% Determine the angle random walk coefficient from the line.
logN = slope*log(1) + b;
N = 10^logN

% Plot the results.
tauN = 1;
lineN = N ./ sqrt(tau);

% Find the index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = 0.5;
logtau = log10(tau);
logadev = log10(adev);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));

% Find the y-intercept of the line.
b = logadev(i) - slope*logtau(i);

% Determine the rate random walk coefficient from the line.
logK = slope*log10(3) + b;
K = 10^logK

% Plot the results.
tauK = 3;
lineK = K .* sqrt(tau/3);

% Find the index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = 0;
logtau = log10(tau);
logadev = log10(adev);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));

% Find the y-intercept of the line.
b = logadev(i) - slope*logtau(i);

% Determine the bias instability coefficient from the line.
scfB = sqrt(2*log(2)/pi);
logB = b - log10(scfB);
B = 10^logB

% Plot the results.
tauB = tau(i);
lineB = B * scfB * ones(size(tau));

tauParams = [tauN, tauK, tauB];
params = [N, K, scfB*B];
figure
loglog(tau, adev, tau, [lineN, lineK, lineB], '--', ...
    tauParams, params, 'o')
title('Allan Deviation of Angular Velocity about X-axis with Noise Parameters')
xlabel('\tau')
ylabel('\sigma(\tau)')
legend('$\sigma (rad/s)$', '$\sigma_N ((rad/s)/\sqrt{Hz})$', ...
    '$\sigma_K ((rad/s)\sqrt{Hz})$', '$\sigma_B (rad/s)$', 'Interpreter', 'latex')
text(tauParams, params, {'N', 'K', '0.664B'})
grid on
axis equal
% Generating a simulated curve
generateSimulatedData = false;

if generateSimulatedData
    % Set the gyroscope parameters to the noise parameters determined
    % above.
    gyro = gyroparams('NoiseDensity', N, 'RandomWalk', K, ...
        'BiasInstability', B);
    omegaSim = helperAllanVarianceExample(L, Fs, gyro);
else
    load('SimulatedSingleAxisGyroscope', 'omegaSim')
end
[avarSim, tauSim] = allanvar(omegaSim, 'octave', Fs);
adevSim = sqrt(avarSim);
adevSim = mean(adevSim, 2); % Use the mean of the simulations.

figure
loglog(tau, adev, tauSim, adevSim, '--')
title('Allan Deviation of HW and Simulation of Angular velocity about X-axis')
xlabel('\tau');
ylabel('\sigma(\tau)')
legend('HW', 'SIM')
grid on
axis equal

theta = cumsum(angv_y2, 1)*t0;
maxNumM = 100;
L = size(theta, 1);
maxM = 2.^floor(log2(L/2));
m = logspace(log10(1), log10(maxM), maxNumM).';
m = ceil(m); % m must be an integer.
m = unique(m); % Remove duplicates.

tau = m*t0;

avar = zeros(numel(m), 1);
for i = 1:numel(m)
    mi = m(i);
    avar(i,:) = sum( ...
        (theta(1+2*mi:L) - 2*theta(1+mi:L-mi) + theta(1:L-2*mi)).^2, 1);
end
avar = avar ./ (2*tau.^2 .* (L - 2*m));
adev = sqrt(avar);
% Find the index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = -0.5;
logtau = log10(tau);
logadev = log10(adev);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));

% Find the y-intercept of the line.
b = logadev(i) - slope*logtau(i);

% Determine the angle random walk coefficient from the line.
logN = slope*log(1) + b;
N = 10^logN

% Plot the results.
tauN = 1;
lineN = N ./ sqrt(tau);

% Find the index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = 0.5;
logtau = log10(tau);
logadev = log10(adev);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));

% Find the y-intercept of the line.
b = logadev(i) - slope*logtau(i);

% Determine the rate random walk coefficient from the line.
logK = slope*log10(3) + b;
K = 10^logK

% Plot the results.
tauK = 3;
lineK = K .* sqrt(tau/3);

% Find the index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = 0;
logtau = log10(tau);
logadev = log10(adev);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));

% Find the y-intercept of the line.
b = logadev(i) - slope*logtau(i);

% Determine the bias instability coefficient from the line.
scfB = sqrt(2*log(2)/pi);
logB = b - log10(scfB);
B = 10^logB

% Plot the results.
tauB = tau(i);
lineB = B * scfB * ones(size(tau));

tauParams = [tauN, tauK, tauB];
params = [N, K, scfB*B];
figure
loglog(tau, adev, tau, [lineN, lineK, lineB], '--', ...
    tauParams, params, 'o')
title('Allan Deviation of Angular Velocity about Y-axis with Noise Parameters')
xlabel('\tau')
ylabel('\sigma(\tau)')
legend('$\sigma (rad/s)$', '$\sigma_N ((rad/s)/\sqrt{Hz})$', ...
    '$\sigma_K ((rad/s)\sqrt{Hz})$', '$\sigma_B (rad/s)$', 'Interpreter', 'latex')
text(tauParams, params, {'N', 'K', '0.664B'})
grid on
axis equal
% Generating a simulated curve
generateSimulatedData = false;

if generateSimulatedData
    % Set the gyroscope parameters to the noise parameters determined
    % above.
    gyro = gyroparams('NoiseDensity', N, 'RandomWalk', K, ...
        'BiasInstability', B);
    omegaSim = helperAllanVarianceExample(L, Fs, gyro);
else
    load('SimulatedSingleAxisGyroscope', 'omegaSim')
end
[avarSim, tauSim] = allanvar(omegaSim, 'octave', Fs);
adevSim = sqrt(avarSim);
adevSim = mean(adevSim, 2); % Use the mean of the simulations.

figure
loglog(tau, adev, tauSim, adevSim, '--')
title('Allan Deviation of HW and Simulation of Angular velocity about Y-axis')
xlabel('\tau');
ylabel('\sigma(\tau)')
legend('HW', 'SIM')
grid on
axis equal

theta = cumsum(angv_z2, 1)*t0;
maxNumM = 100;
L = size(theta, 1);
maxM = 2.^floor(log2(L/2));
m = logspace(log10(1), log10(maxM), maxNumM).';
m = ceil(m); % m must be an integer.
m = unique(m); % Remove duplicates.

tau = m*t0;

avar = zeros(numel(m), 1);
for i = 1:numel(m)
    mi = m(i);
    avar(i,:) = sum( ...
        (theta(1+2*mi:L) - 2*theta(1+mi:L-mi) + theta(1:L-2*mi)).^2, 1);
end
avar = avar ./ (2*tau.^2 .* (L - 2*m));
adev = sqrt(avar);
% Find the index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = -0.5;
logtau = log10(tau);
logadev = log10(adev);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));

% Find the y-intercept of the line.
b = logadev(i) - slope*logtau(i);

% Determine the angle random walk coefficient from the line.
logN = slope*log(1) + b;
N = 10^logN

% Plot the results.
tauN = 1;
lineN = N ./ sqrt(tau);

% Find the index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = 0.5;
logtau = log10(tau);
logadev = log10(adev);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));

% Find the y-intercept of the line.
b = logadev(i) - slope*logtau(i);

% Determine the rate random walk coefficient from the line.
logK = slope*log10(3) + b;
K = 10^logK

% Plot the results.
tauK = 3;
lineK = K .* sqrt(tau/3);

% Find the index where the slope of the log-scaled Allan deviation is equal
% to the slope specified.
slope = 0;
logtau = log10(tau);
logadev = log10(adev);
dlogadev = diff(logadev) ./ diff(logtau);
[~, i] = min(abs(dlogadev - slope));

% Find the y-intercept of the line.
b = logadev(i) - slope*logtau(i);

% Determine the bias instability coefficient from the line.
scfB = sqrt(2*log(2)/pi);
logB = b - log10(scfB);
B = 10^logB

% Plot the results.
tauB = tau(i);
lineB = B * scfB * ones(size(tau));

tauParams = [tauN, tauK, tauB];
params = [N, K, scfB*B];
figure
loglog(tau, adev, tau, [lineN, lineK, lineB], '--', ...
    tauParams, params, 'o')
title('Allan Deviation of Angular Velocity about Z-axis with Noise Parameters')
xlabel('\tau')
ylabel('\sigma(\tau)')
legend('$\sigma (rad/s)$', '$\sigma_N ((rad/s)/\sqrt{Hz})$', ...
    '$\sigma_K ((rad/s)\sqrt{Hz})$', '$\sigma_B (rad/s)$', 'Interpreter', 'latex')
text(tauParams, params, {'N', 'K', '0.664B'})
grid on
axis equal
% Generating a simulated curve
generateSimulatedData = false;

if generateSimulatedData
    % Set the gyroscope parameters to the noise parameters determined
    % above.
    gyro = gyroparams('NoiseDensity', N, 'RandomWalk', K, ...
        'BiasInstability', B);
    omegaSim = helperAllanVarianceExample(L, Fs, gyro);
else
    load('SimulatedSingleAxisGyroscope', 'omegaSim')
end
[avarSim, tauSim] = allanvar(omegaSim, 'octave', Fs);
adevSim = sqrt(avarSim);
adevSim = mean(adevSim, 2); % Use the mean of the simulations.

figure
loglog(tau, adev, tauSim, adevSim, '--')
title('Allan Deviation of HW and Simulation of Angular veloity about Z-axis')
xlabel('\tau');
ylabel('\sigma(\tau)')
legend('HW', 'SIM')
grid on
axis equal