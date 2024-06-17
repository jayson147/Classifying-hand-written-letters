%loading data
load('dataset-letters.mat');

images=(dataset.images);

labels=(dataset.labels);


% 1. Convert the images to double data type
images_double = double(images);

% 2. Display a 3x4 array of figures with random examples
figure; % Creates a new figure
for i = 1:12
    % Select a random image
    idx = randi(size(images_double, 1));
    subplot(3, 4, i); % Creates subplot for each image
    imshow(reshape(images_double(idx, :), [28, 28]), []); % Reshape and display image
    title(char(labels(idx) + 'A' - 1)); % Display label as title converting from ASCII
end

% Save the figure as a PNG file
saveas(gcf, 'random_samples.png');

% 3. Split the dataset into training and testing
numImages = size(images_double, 1);
idx = randperm(numImages); % Random permutation of indices

% Half for training, half for testing
numTrain = round(numImages * 0.5);
trainIdx = idx(1:numTrain);
testIdx = idx(numTrain+1:end);

% Create training and testing datasets
trainImages = images_double(trainIdx, :);
trainLabels = labels(trainIdx);
testImages = images_double(testIdx, :);
testLabels = labels(testIdx);

% 4. Check the label distribution
trainLabelDistribution = histcounts(trainLabels, unique(trainLabels));
testLabelDistribution = histcounts(testLabels, unique(testLabels));


% model training with knn

% Define k
k = 5; 

% Using Euclidean distance

tic;

%training Euclidean Model
knnModelEuclidean = fitcknn(trainImages, trainLabels, 'NumNeighbors', k, 'Distance', 'euclidean');

trainingTimeEuclidean= toc;

tic;
predictedLabelsEuclidean = predict(knnModelEuclidean, testImages);
computeTimeEuclidean = toc;
accuracyEuclidean = sum(predictedLabelsEuclidean == testLabels) / numel(testLabels);

% Using Manhattan distance

tic;
knnModelManhattan = fitcknn(trainImages, trainLabels, 'NumNeighbors', k, 'Distance', 'cityblock');
trainingTimeManhattan=toc;

tic;
predictedLabelsManhattan = predict(knnModelManhattan, testImages);
computeTimeManhattan = toc;
accuracyManhattan = sum(predictedLabelsManhattan == testLabels) / numel(testLabels);

%model training with existing models

% For SVM
tic;
svmModel = fitcecoc(trainImages, trainLabels);
trainingTimeSVM= toc;

tic;
predictedLabelsSVM = predict(svmModel, testImages);
computeTimeSVM = toc;
accuracySVM = sum(predictedLabelsSVM == testLabels) / numel(testLabels);

% For Decision Tree

tic;
treeModel = fitctree(trainImages, trainLabels);
trainingTimeTree= toc; 

tic;
predictedLabelsTree = predict(treeModel, testImages);
computeTimeTree = toc;
accuracyTree = sum(predictedLabelsTree == testLabels) / numel(testLabels);


