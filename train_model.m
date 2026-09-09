close all;
clear all;
rng(0);

data = importdata('Train/labels.txt');
img_nrs = data(:,1);
true_labels = data(:, (2:end));

train_labels = {};
validation_labels = {};
train_patterns = [];
validation_patterns = [];

num_train = 960;
num_validate = 240;

t = tic;
fprintf('Extracting Training Features...\n');
skipped = 0;

for i = 1:num_train
    F = FeatureExtraction(imread(sprintf('Train/captcha_%04d.png', i)));
    num_feats = size(F,1);
    num_true_digits = sum(true_labels(i,:) ~= 0);
    
    if num_feats ~= num_true_digits
        skipped = skipped + 1;
        continue;
    end
    
    if num_feats == 3
        add = 1;
    else
        add = 0;
    end
    
    for j = 1:num_feats
        label = true_labels(i, j+add);
        if label == 0
            continue;
        end
        train_patterns(end+1,:) = squeeze(F(j,:,:));
        train_labels{end+1} = num2str(label);
    end
end

fprintf('Skipped %d images due to digit-count mismatch\n', skipped);
fprintf('Building model...\n');

svm_template = templateSVM(...
    'KernelFunction', 'rbf', ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 3, ...
    'Standardize', true);
    
Mdl = fitcecoc(...
    train_patterns, train_labels, ...
    'Learners', svm_template, ...
    'Coding', 'onevsone');
    
save Mdl
toc(t)

fprintf('\nResubstitution error: %5.2f%%\n\n', 100*resubLoss(Mdl));
train_accuracy = 1 - resubLoss(Mdl);
fprintf('Training accuracy: %5.2f%%\n\n', 100*train_accuracy);

t = tic;
fprintf('Extracting Validation Features...\n');
for i = 1:num_validate
    F = FeatureExtraction(imread(sprintf('Train/captcha_%04d.png', num_train+i)));
    num_feats = size(F,1);
    
    if num_feats == 3
        add = 1;
    else
        add = 0;
    end
    
    for j = 1:num_feats
        label = true_labels(i+num_train, j+add);
        if label == 0
            continue;
        end
        validation_patterns(end+1,:) = squeeze(F(j,:,:));
        validation_labels{end+1} = num2str(label);
    end
end
toc(t)

validation_labels = transpose(validation_labels);
train_labels = transpose(train_labels);

fprintf('Predicting validation set...\n');
t = tic;
validation_pred = predict(Mdl, validation_patterns);
toc(t);

accuracy = mean(cell2mat(validation_pred) == cell2mat(validation_labels));
fprintf('Validation accuracy: %5.2f%%\n', accuracy*100);

f = figure(2);
if (f.Position(3) < 800)
    set(f, 'Position', get(f, 'Position').*[1, 1, 1.5, 1.5]);
end

confusionchart(validation_labels, validation_pred, ...
    'ColumnSummary', 'column-normalized', ...
    'RowSummary', 'row-normalized');
title(sprintf('Validation accuracy: %5.2f%%\n', accuracy*100));