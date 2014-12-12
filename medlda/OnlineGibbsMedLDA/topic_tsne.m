clear;
load('topics.mat');
for time = 2:23
    allmat = [];
    for ci = 1:size(topics,2)
        mat = topics(time, ci, :, :);
        mat = reshape(mat, size(topics,3), size(topics,4));
        mat_mean = repmat(mean(mat), size(mat,1), 1);
        allmat = [allmat mat];
    end
    ydata = tsne(allmat, labels, 2);
    save(1, ['topic_' num2str(time) '.png']);
end