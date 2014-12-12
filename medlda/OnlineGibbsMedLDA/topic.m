load('topics.mat');
figure(1);
filename = 'topic.gif';
ci = 10;
mat = topics(1, ci, :, :);
mat = reshape(mat, size(topics,3), size(topics,4));
mat_mean = repmat(mean(mat), size(mat,1), 1);
mat = topics(end, ci, :, :);
mat = reshape(mat, size(topics,3), size(topics,4));
mat = mat - mat_mean;
corr = mat' * mat;
[U, S, V] = svd(corr);
for time = 1:size(topics,1)
    mat = topics(time, ci, :, :);
    mat = reshape(mat, size(topics,3), size(topics,4));
    mat = mat - mat_mean;

    coef = [mat * U(:,1), mat * U(:,2)];
    plot(coef(labels~=ci, 1), coef(labels~=ci, 2), 'b.'); hold on;
    plot(coef(labels==ci,1), coef(labels==ci, 2), 'r.'); 
    axis([-1, 1, -1, 1]);
    hold off;
    pause;
end