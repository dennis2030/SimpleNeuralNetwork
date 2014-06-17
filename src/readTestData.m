function [images, labels, numOfImages] = readTestData()

% read config json
[data, json] = parse_json(fileread('train.json'));

activateType = data{1}.activateType;
test_image_fname = data{1}.test_data;
test_labels = data{1}.test_label;
numOfLayers = data{1}.numOfLayers;

% load test data && labels
images = loadMNISTImages(test_image_fname);
labels = loadMNISTLabels(test_labels);
numOfImages = size(images,2);

end
