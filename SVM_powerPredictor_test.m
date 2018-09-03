tic

wLen = 5*dLen; %input window length
%wLen = 10;
N_train = dLen*N/2-dLen+1; %training data length
N_test = dLen*N/2; %test data length
N = N_train+N_test; %total data length (THIS N IS DIFFERENT THAN N IN ARP_simulator.m)

%Training label ground truth/target
trainLbl = real(totalAvgPwr(wLen+1:N_train));

trainDatas = zeros(wLen,N_train-wLen);
%Traing and test input data
trainData = zeros(wLen,N_train-wLen);
testData = zeros(wLen,N_test-wLen);

for i = wLen:N_train-1
    %Input consists of present state and wLen previous states
    trainData(:,i-wLen+1) = real(totalAvgPwr(i-wLen+1:i));
end

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
predAcc = sum(repmat(predState_1step,dLen,1)==trueState,2)/(N_test-wLen);
%coherent
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

%Sliding observed accuracies
%dominant state
predAcc_obs1 = (sum(predState_1step(ambState==0) == trueState(1,ambState...
    ==0)) + sum(predState_1step(ambState==1) == (sum(trueState(:,...
    ambState==1)) >= dLen/2)))/(N_test-wLen);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%strict signal state
predAcc_obs2 = (sum(predState_1step(ambState==0) == trueState(1,ambState...
    ==0)) + sum(predState_1step(ambState==1) == (sum(trueState(:,...
    ambState==1)) > 0)))/(N_test-wLen);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%strict signal state
predAcc_obs2_coh(1) = (sum(predState_1step(1:dLen:N_test-wLen)...
    == (sum(trueState(:,1:dLen:N_test-wLen)) > 0)))/(N_test-wLen)*dLen;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %strict signal state
    predAcc_obs2_coh(i+1) = (sum(predState_1step(1+i:dLen:N_test-wLen-i)...
        == (sum(trueState(:,1+i:dLen:N_test-wLen-i)) > 0)))/(N_test-wLen)*dLen;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %true energy detector state
    predAcc_obs3_coh(i+1) = (sum(predState_1step(ambState(1+i:dLen:N_test-wLen-i)==0)...
        == trueState(1,ambState(1+i:dLen:N_test-wLen-i)==0)) + sum(...
        predState_1step(ambState(1+i:dLen:N_test-wLen-i)==1) == (obsState(...
        predPwrStates(1,ambState(1+i:dLen:N_test-wLen-i)==1)))))/(N_test-wLen)*dLen;
end


%Plot one-step ahead prediction with threshold

figure
plot(1:N_test-wLen,10*log10(abs(real(totalAvgPwr(N_train+1+wLen:N))))-30,...
1:N_test-wLen,10*log10(abs(score))-30,1:N_test-wLen,10*log10(thresh*ones(1,N_test-wLen))-30)
legend('Input Signal','Prediction')
xlabel('Samples')
ylabel('Magnitude (dBm)')
title('one-step ahead prediction')



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

figure
plot(N_train+wLen+1:N-fSteps+1,10*log10(abs(totalAvgPwr(...
    N_train+wLen+1:N-fSteps+1)))-30,N_train+wLen+1:N-fSteps+1,...
    10*log10(abs(score(1,:)))-30,N_train+wLen+1:N-fSteps+1,...
    10*log10(ones(1,length(predPwrStates))*thresh)-30)


toc