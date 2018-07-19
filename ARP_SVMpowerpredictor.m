%Jacob Kovarskiy%

clear all
close all

lambda1 = 0.8; %on-time decay rate per sample (1s)
lambda2 = 0.8; %off-time decay rate per sample (0s)
kParam1 = 2; %k-parameter for Erlang/gamma distribution (ON)
kParam2 = 2; %k-parameter for Erlang/gamma distribution (OFF)
var1 = lambda1; %variance parameter for log-normal distribution (ON)
var2 = lambda2; %variance parameter for log-normal distribution (OFF)
N = 300; %number of samples
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
%fileID = fopen('svmDataSet.txt','w+');
%fprintf(fileID,totalAvgPwr.');
csvwrite('svmDataSet.csv',totalAvgPwr.');
csvwrite('svmDataSet.txt',totalAvgPwr.');
%Observed states based on energy detector
obsState = totalAvgPwr > thresh;

