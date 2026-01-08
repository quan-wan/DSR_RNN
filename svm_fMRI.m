%% fMRI - within- and cross-label decoding for context
%conditions=[PMI_ori;UMI_ori;PMI_loc;UMI_loc;Cue1;ori_1,ori_2,loc_1,loc_2,cue_switch,PMI_ori_2,UMI_ori_2,PMI_loc_2,UMI_loc_2];

% clear all
sublist = {'02','03','04','07','08','09','10','11','12','16','17','18','19'};
for roi = 1:8
    for sub = 1:length(sublist)
        filenames = {'sample_500_roi1and2','delay2_500_roi18','delay2_500_roi19','delay2_500_roi20','delay2_500_roi21','delay2_500_roi22' ...
            'delay2_500_roi23','delay2_500_roi25'};
        load (['POCONN4_' sublist{sub}  '_all_trials_' filenames{roi} '.mat'])
        for tp = 1:20
            tp_data = squeeze(trial_pat(:,:,tp));

            %define category labels
            Y =categorical(conditions(8,:));
            classOrder=unique(Y);
            rng(10,'twister');
            t = templateSVM('Standardize',1,'BoxConstraint',5);
            Mdl = fitcecoc(tp_data,Y,'Learners',t,'ClassNames',classOrder,'Coding','onevsone');

            c = cvpartition(Y,'KFold',10,'Stratify',true);

            CVMdl = crossval(Mdl,'CVPartition',c);

            predicted_label = kfoldPredict(CVMdl);
            Y2 = categorical(conditions(9,:));

            acc_stim1to1(sub,tp)=sum((predicted_label'== Y))/length(Y); %cross-validation
            acc_stim1to2(sub,tp)=sum((predicted_label'== Y2))/length(Y2);  %cross-decoding
            tp
        end
        sub
    end
    save(['C:\Users\cogsw\OneDrive\Desktop\roi_new_with_cue1_type\context_1v1_C5\trainStim1_' filenames{roi} '.mat'],'acc_stim1to1','acc_stim1to2')
    clearvars -except roi sublist sub
end



%% fMRI - within- and cross-label decoding for priority (flip labels at TR15 for switch trials)
%conditions=[PMI_ori;UMI_ori;PMI_loc;UMI_loc;Cue1;ori_1,ori_2,loc_1,loc_2,cue_switch,PMI_ori_2,UMI_ori_2,PMI_loc_2,UMI_loc_2];
clear all
sublist = {'02','03','04','07','08','09','10','11','12','16','17','18','19'};
for roi = 1:8
    for sub = 1:length(sublist)
        filenames = {'sample_500_roi1and2','delay2_500_roi18','delay2_500_roi19','delay2_500_roi20','delay2_500_roi21','delay2_500_roi22' ...
            'delay2_500_roi23','delay2_500_roi25'};
        load (['POCONN4_' sublist{sub}  '_all_trials_' filenames{roi} '.mat'])
        for tp = 1:20
            tp_data = squeeze(trial_pat(:,:,tp));

            %define category labels
            if tp <= 14 
                Y =categorical(conditions(3,:));
            else
                Y =categorical(conditions(13,:));
            end
            classOrder=unique(Y);
            rng(10,'twister');
            t = templateSVM('Standardize',1,'BoxConstraint',5);
            Mdl = fitcecoc(tp_data,Y,'Learners',t,'ClassNames',classOrder,'Coding','onevsone');

            c = cvpartition(Y,'KFold',10,'Stratify',true);

            CVMdl = crossval(Mdl,'CVPartition',c);

            predicted_label = kfoldPredict(CVMdl);
            if tp <= 14
                Y2 = categorical(conditions(4,:));
            else
                Y2 = categorical(conditions(14,:));
            end
            
            acc_pmi_pmi(sub,tp)=sum((predicted_label'== Y))/length(Y);
            acc_pmi_umi(sub,tp)=sum((predicted_label'== Y2))/length(Y2);
            tp
        end
        sub
    end
    save(['C:\Users\cogsw\OneDrive\Desktop\roi_new_with_cue1_type\priority_1v1_C5\trainPMI_' filenames{roi} '.mat'],'acc_pmi_pmi','acc_pmi_umi')
    clearvars -except roi sublist sub
end

%% average the columns (within- and cross-decoding), run t-test for each timepoint and plotting

%plot accuracy for priority
acc_crossval = (acc_pmi_pmi + acc_umi_umi)/2;
acc_crossdec = (acc_pmi_umi + acc_umi_pmi)/2;

%plot accuracy for context
% acc_crossval = (acc_stim1to1 + acc_stim2to2)/2;
% acc_crossdec = (acc_stim1to2 + acc_stim2to1)/2;

acc_allsub = acc_crossval;

for tp = 1:20
[h(tp),p(tp),ci{tp},stats{tp}] = ttest(acc_allsub(:,tp),1/9,'tail','right');
end
[adj_h, crit_p, adj_ci_cvrg, adj_p]=fdr_bh(p);


acc_avgsub = mean(acc_allsub,1);
tpnum = size(acc_allsub,2);

times = 1:20;
options.handle = figure('DefaultAxesFontSize',14);

options.color_area = [128 193 219]./255;    % Blue theme
options.color_line = [ 52 148 186]./255;

options.x_axis = times;
options.alpha      = 0.5;
options.line_width = 2;
options.error      = 'sem';
fig = plot_areaerrorbar(acc_allsub,options);

hold on
plot(times,ones(1,tpnum)*0.1111,'k--','LineWidth',1.3);

adj_h = double(adj_h);
hplot = adj_h;
hplot(adj_h==0) = NaN;
plot(times,hplot*0.107,'ks','MarkerSize',10,'MarkerFaceColor','r');
xlabel('TR');
ylabel('decoding accuracy')
% ylim([0., 0.13])
xticks(0:2:20)
