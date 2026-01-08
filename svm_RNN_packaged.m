%% Prepare the data, add variables of PMI and UMI
for rnn = 1:5
    netname = {'20240113132728','20240111122909', '20240111122007', '20240111121514', '20240111120911'};
    load(['Noise001JitterPlus10_relu_' netname{rnn} '-Copy1_324_CB.mat'])

    %generate pmi and umi labels based on stim and cues
    for tr = 1:324
        if cues1(tr) == 1
            pmi(tr) = stims1(tr);
            umi(tr) = stims2(tr);
        elseif cues1(tr) == -1
            pmi(tr) = stims2(tr);
            umi(tr) = stims1(tr);
        end

        if cues2(tr) == 1
            pmi2(tr) = stims1(tr);
            umi2(tr) = stims2(tr);
        elseif cues2(tr) == -1
            pmi2(tr) = stims2(tr);
            umi2(tr) = stims1(tr);
        end

    end
    hidden = permute(hidden,[3 2 1]);

    % Run SVM classifier
    acc_pmi = zeros(350,4);
    acc_umi = zeros(350,4);
    acc_stims1 = zeros(350,4);
    acc_stims2 = zeros(350,4);

    parfor run = 1:4
        tmp = zeros(350,324);
        for tp = 1:350
            data = squeeze(hidden(:,tp,:));

            %switch PMI label upon Cue 2
            if tp <= 300
                Y = categorical(pmi);
                Y2 = categorical(umi);
            else
                Y = categorical(pmi2);
                Y2 = categorical(umi2);
            end
            Z = categorical(stims1);
            Z2 = categorical(stims2);

            %set training label
            switch run
                case 1
                    train_label = Y; %train on PMI label
                case 2
                    train_label = Y2; %train on UMI label
                case 3
                    train_label = Z; %train on Sample 1 label
                case 4
                    train_label = Z2; %train on Sample 2 label
            end

            classOrder=unique(Z2);
            rng(10,'twister');
            t = templateSVM('Standardize',1,'BoxConstraint',5);

            Mdl = fitcecoc(data',train_label,'Learners',t,'ClassNames',classOrder,'Coding','onevsone');

            c = cvpartition(train_label,'KFold',10,'Stratify',true);
            CVMdl = crossval(Mdl,'CVPartition',c);
            predicted_label = kfoldPredict(CVMdl);
            tmp(tp,:) = double(predicted_label);

            acc_pmi(tp,run)=sum((predicted_label'==Y))/length(Y);
            acc_umi(tp,run)=sum((predicted_label'==Y2))/length(Y2);
            acc_stims1(tp,run)=sum((predicted_label'==Z))/length(Z);
            acc_stims2(tp,run)=sum((predicted_label'==Z2))/length(Z2);
            tp

        end
    end
    save(['rnn_' netname{rnn} '_C5.mat'],'acc_pmi','acc_umi','acc_stims1','acc_stims2')
end

%% Plotting
tp_num = 350;
figure('DefaultAxesFontSize',14);
hold on
times = 1:tp_num;
plot(times,acc_stims1,'LineWidth',3)
plot(times,ones(1,tp_num)*0.1111,'k--','LineWidth',1.5);
ylim([0 1])
xlim([0,tp_num])
xticks(0:50:tp_num)
xlabel('timestep');
ylabel('decoding accuracy')
% title('train stim2 test stim1')

figure('DefaultAxesFontSize',14);
hold on
plot(times,acc_stims2,'LineWidth',3)
plot(times,ones(1,tp_num)*0.1111,'k--','LineWidth',1.5);
ylim([0 1])
xlim([0,tp_num])
xticks(0:50:tp_num)
xlabel('timestep');
ylabel('decoding accuracy')
% title('train stim2 test stim2')