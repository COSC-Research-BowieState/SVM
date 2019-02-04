clear all
close all

lambda1 = 0.8; %on-time decay rate per sample (1s)
lambda2 = 0.8; %off-time decay rate per sample (0s)
kParam1 = 2; %k-parameter for Erlang/gamma distribution (ON)
kParam2 = 2; %k-parameter for Erlang/gamma distribution (OFF)
var1 = lambda1; %variance parameter for log-normal distribution (ON)
var2 = lambda2; %variance parameter for log-normal distribution (OFF)
N = 30000; %number of samples
occupancy = zeros(1,N);
stateTrans = [];
intTimes = [];
intTimesSeq = [];
upDist = 'lnorm'; %'exp', 'erl', or 'lnorm'
downDist = 'lnorm'; %'exp', 'erl', or 'lnorm'

%Process initialized to "on"
%Exponential distribution during on times 
%Off times can switch between exponential, k-erlang, and log normal

totalTime = 0; %Tracks total time of generated by ARP
seqState = 1; %Tracks next state to generate

%
while totalTime < N
    %Generates on sequence
    if seqState 
        %generates random on period
        switch upDist
            case 'exp'
                period = ceil(exprnd(1/lambda1));
            case 'erl'
                period = ceil(gamrnd(kParam1,1/lambda1)); %assumes k=2
            case 'lnorm'
                trueMu = log(((1/lambda1)^2)/sqrt((1/var1)+(1/lambda1)^2));
                trueSig = sqrt(log((1/var1)/((1/lambda1)^2)+1));
                period = ceil(lognrnd(trueMu,trueSig)); %assumes mean=var=lambda
        end
        %period = 5; %Uncomment this to make deterministic
        if (totalTime + period) > N %makes sure total time isn't exceeded
            occupancy(totalTime+1:N) = ones(1,N-totalTime);
        else %appends the proper sequence of 1s
            occupancy(totalTime+1:totalTime+period) = ones(1,period);
        end
        
        %tracks state transitions and on/off durations
        stateTrans = [stateTrans 1];
        intTimes = [intTimes period];
        intTimesSeq = [intTimesSeq 1:period];
        seqState = 0;
        
    %Generates off sequence
    else      
        %generates random off period
        switch downDist
            case 'exp'
                period = ceil(exprnd(1/lambda2));
            case 'erl'
                period = ceil(gamrnd(kParam2,1/lambda2)); %assumes k=2
            case 'lnorm'
                period = ceil(lognrnd(log(((1/lambda2)^2)/...
                    sqrt((1/var2)+(1/lambda2)^2)),...
                    sqrt(log((1/var2)/((1/lambda2)^2)+1)))); %assumes mean=var=lambda
        end
        %period = 10; %Uncomment this to make deterministic
        if (totalTime + period) > N %makes sure total time isn't exceeded
            occupancy(totalTime+1:N) = zeros(1,N-totalTime);
        else %appends the proper sequence of 0s
            occupancy(totalTime+1:totalTime+period) = zeros(1,period);
        end
        
        %tracks state transitions and on/off durations
        stateTrans = [stateTrans 0];
        intTimes = [intTimes period];
        intTimesSeq = [intTimesSeq 1:period];
        seqState = 1;
        
    end
    
    totalTime = totalTime + period;
    
end
%}

seqSize = length(stateTrans); %total number of on and off states
traffic_intensity = mean(occupancy>0); %measures traffic intenisty
%measures mean signal interarrival
mean_int = sum(intTimes(1:seqSize-mod(seqSize,2)))/...
    ((seqSize-mod(seqSize,2))/2);
actual_int = 1/lambda1+1/lambda2; %calculates theoretical interarrival

upTimes = intTimes(stateTrans==1); %tracks durations of up times
downTimes = intTimes(stateTrans==0); %tracks durations of down times

%Transition detector "accuracy/error"
predicted = occupancy(1:N-1);
%Theoretical accuracy based on lambda parameters
expected_i_guess = 1-(2/actual_int-1/N)
%Accuracy based on measured mean interarrival
other_i_guess = 1-(2/mean_int-1/N)
%Observed accuracy
accuracy_i_guess = sum(predicted==occupancy(2:N))/(N-1)


%
%%%%%%%%%input RF signal generation%%%%%%%%%
dLen = 100; %length of the energy detector
fs = 100e6;
time = linspace(0,N*dLen/fs,N*dLen);
powerLvl = -40; %power in dBm
amp = sqrt((10^(powerLvl/10))/1000*(2*50)); %sinusoid amplitude
noiseVar = 1e-7; %noisefloor variance (1e-6 places noisefloor around -100 dBm)
noisefloor = sqrt(noiseVar)*randn(1,N*dLen);

sineWave = amp*exp(1j*2*pi*10e6*time); %Sine wave at 10 MHz
%Average SNR of signal
SNR = 10*log10((sum(abs(sineWave).^2)/(dLen*N))/(sum(abs(noisefloor).^2)...
    /(dLen*N)));

%Modulates sine wave with occupancy state where each state has dLen samples
occSwitch = reshape(repmat(occupancy,dLen,1), [1, N*dLen]);
inputRF = sineWave.*occSwitch+noisefloor;
%figure
%plot(linspace(-50,50,N*dLen),10*log10(abs(fftshift(fft(sineWave)/(dLen*N))).^2)+10)

P_fa = 0.01; %probability of false alarm
%energy detector threshold
thresh = noiseVar/sqrt(dLen)*qfuncinv(P_fa)+noiseVar; 

%Calculates total average power over a sliding window
totalAvgPwr = zeros(1,dLen*N-dLen+1);
pwrStates = zeros(dLen, dLen*N-dLen+1);
for i=1:dLen*N-dLen+1
    totalAvgPwr(i) = sum(abs(inputRF(i:i+dLen-1)).^2)/dLen;
    pwrStates(:,i) = [i:i+dLen-1];
end

%Observed states based on energy detector
obsState = totalAvgPwr > thresh;

%Plots total average power and detection threshold
figure
subplot(2,1,1)
t = 1:dLen*N-dLen+1
%disp(t);
%disp(class(t))
plot(t,10*log10(totalAvgPwr)-30,1:dLen*N-dLen+1,10*log10(thresh*ones(size(totalAvgPwr)))-30)
ylabel('Total Average Power (dBm)')
xlabel('Samples')
%subplot(2,1,2)
plot(real(inputRF(dLen:dLen*N)))
xlabel('Samples')
ylabel('Amplitude (V)')

%Calculates detection accuracy, false alarm rate, and missed detection rate
%detection accuracy evaluated in terms of soonest detection
dAcc = sum(obsState==occSwitch(dLen:dLen*N))/(dLen*N-dLen+1);
%coherent detection accuracy with the window jumping by dLen samples
%per observations
dAcc_coherent = sum(obsState(1:dLen:dLen*N-dLen+1)==occupancy)/N;
faRate = sum(obsState.*(~occSwitch(dLen:dLen*N)))/(dLen*N-dLen+1);
mdRate = sum(~(obsState).*occSwitch(dLen:dLen*N))/(dLen*N-dLen+1);

%}
tic
dLen = 10;
wLen = 5*dLen; %input window length
%wLen = 10;
%N = 30;
N_train = dLen*N/2-dLen+1; %training data length
N_test = dLen*N/2; %test data length
N = N_train+N_test; %total data length

%Training label ground truth/target
trainLbl = real(totalAvgPwr(wLen+1:N_train));

%Traing and test input data
trainData = zeros(wLen,N_train-wLen);
testData = zeros(wLen,N_test-wLen);

for i = wLen:N_train-1
    %Input consists of present state and wLen previous states
    trainData(:,i-wLen+1) = real(totalAvgPwr(i-wLen+1:i));
end
disp = wLen:N_test-1;

for i = wLen:N_test-1
    %Input consists of present state and wLen previous states
    testData(:,i-wLen+1) = real(totalAvgPwr(i-wLen+1+N_train:i+N_train));
end
predPwrStates = pwrStates(:,wLen+1+N_train:N);

%trains the SVM
theSVM = fitrsvm(trainData',trainLbl','KernelFunction','linear',...
    'Standardize','on','KernelScale','auto','Solver','ISDA');

toc

fSteps = dLen; %tracks number of future steps to predict
predicted = zeros(fSteps, N_test-wLen);

%cyclically adds previous predicted sample as input to predict next sample
%repeated until fSteps samples ahead are predicted
nData = testData;
for i=1:5
    predicted(i,:) = predict(theSVM, nData')';
    nData = [testData(2:wLen,:); predicted(i,:)];
end

toc

%Rearranges fSteps-ahead prediction vectors so predictions don't overlap
%results in fSteps sets of predicted sequences
predSet = zeros(fSteps, N_test-wLen);
setCnt = 1;
obsSample = zeros(1,N_test-wLen);
for i=1:N_test-wLen
    predSet(setCnt,i-setCnt+1:i-setCnt+fSteps) = predicted(:, i)';
    if setCnt==fSteps
        obsSample(i-setCnt+1) = 1;
        setCnt = 1;
    else
        setCnt = setCnt + 1;
    end
end

score = predicted(1,:); %one step ahead prediction

%prediction_accuracy = sum(classOut==testLbl')/(N_test-wLen)

%function fitting
%measure one-step ahead MSE
MSE_onestep = sum((real(totalAvgPwr(N_train+1+wLen:N))-score).^2)/(N_test-wLen);
figure
plot(1:N_test-wLen,10*log10(abs(real(totalAvgPwr(N_train+1+wLen:N))))-30,...
    1:N_test-wLen,10*log10(abs(score))-30)
legend('Input Signal','Prediction')
xlabel('Samples')
ylabel('Magnitude (dBm)')
title('one-step ahead prediction')

MSE_multistep = zeros(fSteps,1);
for i=1:fSteps
    %measures multistep MSE
    MSE_multistep(i) = sum((real(totalAvgPwr(N_train+i+wLen:N-fSteps+i))-predSet(i,...
    1:N_test-wLen-fSteps+1)).^2)/(N_test-wLen-fSteps+1);
    
    %plots first five sets of predicted sequences
    if i<=5
        figure
        timestep = i:N_test-wLen-fSteps+i;
        %{
        plot(timestep,real(inputRF(N_test+i+wLen:N-fSteps+i)),...
            timestep,predSet(i,1:N_test-wLen-fSteps+1),...
            timestep(obsSample(1:N_test-wLen-fSteps+1)==1),...
            predSet(i,obsSample(1:N_test-wLen-fSteps+1)==1),'xk')
        %}
        plot(timestep,10*log10(abs(real(totalAvgPwr(N_train+i+wLen:N-...
            fSteps+i))))-30,...
            timestep,10*log10(abs(predSet(i,1:N_test-wLen-fSteps+1)))-30)
        legend('Input Signal','Prediction')
        xlabel('Samples')
        ylabel('Magnitude (dBm)')
        title('multi-step ahead prediction')
    end
end

P_fa = 0.01; %probability of false alarm
%energy detector threshold
thresh = noiseVar/sqrt(dLen)*qfuncinv(P_fa)+noiseVar;

%{
%measures the total average power for each predicted sequence
predAvgPwr = zeros(fSteps,length(predSet)-fSteps+1);
for i = 1:length(predSet)-fSteps+1
    predAvgPwr(:,i) = sum(abs(predSet(:,i:i+dLen-1)).^2,2)/dLen;
end

%thresh = 6.5*thresh; %raises the threshold to account for prediction error
predState = predAvgPwr > thresh; 
%}
thresh = 1*thresh; %raises the threshold to account for prediction error
predState_1step = score > thresh;

%Single sample state accuracy (lol dumb)
trueState = occSwitch(predPwrStates);
%sliding
%predAcc = sum(repmat(predState_1step,dLen,1)==trueState,2)/(N_test-wLen);
%{
coherent
predAcc_coh = sum(repmat(predState_1step(1:dLen:N_test-wLen),dLen,1)==...
    trueState(:,1:dLen:N_test-wLen),2)/(N_test-wLen)*dLen;

%Unambiguous accuracy vs ambiguous accuracy (sliding)
UA_cnt = 0;
A_cnt = 0;
predAcc_UA = 0;
predAcc_A = zeros(dLen,1);
ambState = zeros(1,N_test-wLen);
for i=1:N_test-wLen
    if sum(trueState(:,i)==ones(dLen,1))==5 ||...
            sum(trueState(:,i)==zeros(dLen,1))==5
        predAcc_UA = predAcc_UA + (predState_1step(i)==trueState(1,i));
        UA_cnt = UA_cnt + 1;
    else
        predAcc_A = predAcc_A + (predState_1step(i)==trueState(:,i));
        A_cnt = A_cnt + 1;
        ambState(i) = 1;
    end
end
predAcc_UA_tot = predAcc_UA/UA_cnt;
predAcc_A_tot = predAcc_A/A_cnt;

A_idx = zeros(1,A_cnt);
re_A = 1;
UA_idx = zeros(1,UA_cnt);
re_UA = 1;
for i=1:N_test-wLen
    if ambState(i)
        A_idx(re_A) = i;
        re_A = re_A + 1;
    else
        UA_idx(re_UA) = i;
        re_UA = re_UA + 1;
    end
end
%}
%Sliding observed accuracies
%dominant state
predAcc_obs1 = (sum(predState_1step(ambState==0) == trueState(1,ambState...
    ==0)) + sum(predState_1step(ambState==1) == (sum(trueState(:,...
    ambState==1)) >= dLen/2)))/(N_test-wLen);

%strict signal state
predAcc_obs2 = (sum(predState_1step(ambState==0) == trueState(1,ambState...
    ==0)) + sum(predState_1step(ambState==1) == (sum(trueState(:,...
    ambState==1)) > 0)))/(N_test-wLen);

%true energy detector state
predAcc_obs3 = (sum(predState_1step(ambState==0) == trueState(1,ambState...
    ==0)) + sum(predState_1step(ambState==1) == (obsState(predPwrStates(...
    1,ambState==1)))))/(N_test-wLen);

predAcc_obs1_coh = zeros(dLen,1);
predAcc_obs2_coh = zeros(dLen,1);
predAcc_obs3_coh = zeros(dLen,1);

%Coherent observed accuracies
%dominant state
predAcc_obs1_coh(1) = (sum(predState_1step(ambState(1:dLen:N_test-wLen)==0)...
    == trueState(1,ambState(1:dLen:N_test-wLen)==0)) + sum(...
    predState_1step(ambState(1:dLen:N_test-wLen)==1) == (sum(trueState(:,...
    ambState(1:dLen:N_test-wLen)==1)) >= dLen/2)))/(N_test-wLen)*dLen;

%strict signal state
predAcc_obs2_coh(1) = (sum(predState_1step(1:dLen:N_test-wLen)...
    == (sum(trueState(:,1:dLen:N_test-wLen)) > 0)))/(N_test-wLen)*dLen;

%true energy detector state
predAcc_obs3_coh(1) = (sum(predState_1step(ambState(1:dLen:N_test-wLen)==0)...
    == trueState(1,ambState(1:dLen:N_test-wLen)==0)) + sum(...
    predState_1step(ambState(1:dLen:N_test-wLen)==1) == (obsState(...
    predPwrStates(1,ambState(1:dLen:N_test-wLen)==1)))))/(N_test-wLen)*dLen;

for i=1:dLen-1
    %dominant state
    predAcc_obs1_coh(i+1) = (sum(predState_1step(ambState(1+i:dLen:N_test-wLen-i)==0)...
        == trueState(1,ambState(1+i:dLen:N_test-wLen-i)==0)) + sum(...
        predState_1step(ambState(1+i:dLen:N_test-wLen-i)==1) == (sum(trueState(:,...
        ambState(1+i:dLen:N_test-wLen-i)==1)) >= dLen/2)))/(N_test-wLen)*dLen;

    %strict signal state
    predAcc_obs2_coh(i+1) = (sum(predState_1step(1+i:dLen:N_test-wLen-i)...
        == (sum(trueState(:,1+i:dLen:N_test-wLen-i)) > 0)))/(N_test-wLen)*dLen;

    %true energy detector state
    predAcc_obs3_coh(i+1) = (sum(predState_1step(ambState(1+i:dLen:N_test-wLen-i)==0)...
        == trueState(1,ambState(1+i:dLen:N_test-wLen-i)==0)) + sum(...
        predState_1step(ambState(1+i:dLen:N_test-wLen-i)==1) == (obsState(...
        predPwrStates(1,ambState(1+i:dLen:N_test-wLen-i)==1)))))/(N_test-wLen)*dLen;
end


%Plot one-step ahead prediction with threshold
%{
figure
plot(1:N_test-wLen,10*log10(abs(real(totalAvgPwr(N_train+1+wLen:N))))-30,...
1:N_test-wLen,10*log10(abs(score))-30,1:N_test-wLen,10*log10(thresh*ones(1,N_test-wLen))-30)
legend('Input Signal','Prediction')
xlabel('Samples')
ylabel('Magnitude (dBm)')
title('one-step ahead prediction')
%}


%{
occSwitch1 = zeros(size(predState));
occSwitch2 = zeros(fSteps,length(N_train+wLen+1:dLen:N-fSteps+1));
%Rearranges the occupancy states to align with each prediction sequence
for i=1:fSteps
   %aligns states for soonest detection with a sliding window
   occSwitch1(i,:) = occSwitch(N_train+i+wLen:N-fSteps+i);
   %aligns states for coherent detection with a staggered window
   occSwitch2(i,:) = occSwitch(N_train+wLen+i:dLen:N-fSteps+i);
end

%soonest detection accuracy
state_prediction_accuracy1 = sum(predState == occSwitch1,2)/...
    (N_test-wLen-fSteps+1);

%coherent detection accuracy
state_prediction_accuracy2 = sum(predState(:,1:dLen:length(predSet)-dLen+...
    1)==occSwitch2,2)/(length(predSet)/fSteps);
%}

%{
pred_faRate = sum(predState.*(~occSwitch(N_test+1+wLen:N-fSteps+1)))/...
    (N_test-wLen-fSteps+1);
pred_mdRate = sum(~(predState).*occSwitch(N_test+1+wLen:N-fSteps+1))/...
    (N_test-wLen-fSteps+1);
%}

%plots actual total average power vs. predicted power
%{
figure
plot(N_train+wLen+1:N-fSteps+1,10*log10(abs(totalAvgPwr(...
    N_train+wLen+1:N-fSteps+1)))-30,N_train+wLen+1:N-fSteps+1,...
    10*log10(abs(score(1,:)))-30,N_train+wLen+1:N-fSteps+1,...
    10*log10(ones(1,length(predPwrStates))*thresh)-30)
%}

toc