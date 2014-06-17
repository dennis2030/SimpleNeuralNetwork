function test()

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

% load trained NN
load 'NN.mat';
right_guess = 0;
guess = zeros(10,1);

for i=1:numOfImages
    NN(1).a = images(:,i);
    NN = forward_prop(NN, numOfLayers, activateType);
    [value, idx] = max(NN(numOfLayers).a);
    if(idx == labels(i)+1)
        right_guess = right_guess + 1;
    end
    guess(idx) = guess(idx) + 1;

end

disp([num2str((right_guess/numOfImages)*100.0) '%'])
