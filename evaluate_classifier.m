tic
data = importdata('Train/labels.txt');
img_nrs = data(:,1);
true_labels = data(:, (2:end));

my_labels = zeros(size(true_labels));
N = size(img_nrs,1);

for n = 1:N
    k = img_nrs(n);
    fprintf('Processing captcha %04d\n', k);
    im = imread(sprintf('Train/captcha_%04d.png', k));
    my_labels(k,:) = myclassifier(im);
end

correct = sum(abs(true_labels - my_labels),2) == 0;
fprintf('\n\nAccuracy: \n');
fprintf('%f\n\n', mean(correct));
toc