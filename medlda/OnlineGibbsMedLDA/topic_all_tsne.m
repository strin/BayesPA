clear;
load('topics.mat');
allmat = [];
alllabel = [];
Y = [];
step = 5;
ind = 1:step:size(topics,3);
for time = 23:-1:1
    mats = [];
    alllabel = [alllabel; labels(ind)'];
    for ci = 1:size(topics, 2)
        mat = topics(time, ci, ind, :);
        mat = reshape(mat, length(ind), size(topics,4));
        mats = [mats mat];
    end
    if time == size(topics,1)
        %corr = X' * X;
        %[U, S, V] = svd(corr);
        %Y = [X * U(:,1), X * U(:,2)]; 
      % mat_mean = repmat(mean(mats), size(mats,1), 1);
      Y = tsne(mats, labels(ind), 2);
    else
    % X = mats - mat_mean;
      Y = tsne(mats, labels(ind), Y);
    end
    saveas(1, ['topic_' num2str(time) '.png']);
end
