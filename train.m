function train()

% read config json
[data, json] = parse_json(fileread('train.json'));
max_iter = data{1}.max_iter
pretrain_iter = data{1}.pretrain_iter
numOfLayers = data{1}.numOfLayers;
weight_decay = data{1}.weight_decay
epsilon = data{1}.epsilon
learning_rate = data{1}.learning_rate
Layers = data{1}.Layers
activateType = data{1}.activateType
train_image_fname = data{1}.train_data
train_labels = data{1}.train_label
batchSize = data{1}.batchSize
preTrain = data{1}.preTrain
beta = data{1}.beta
lo = data{1}.lo
preTrain_batchSize = data{1}.preTrain_batchSize
learning_rate_decay = data{1}.learning_rate_decay
momentum = data{1}.momentum;

% read input data && label
images = loadMNISTImages(train_image_fname);
labels = loadMNISTLabels(train_labels);
numOfImages = size(images,2);

[testImages, testLabels, test_numOfImages] = readTestData();

% initialize layer info
NN = initialNN(numOfLayers, Layers, epsilon);
counter = 0;

% pre_training part

% initialization
trained_w = cell(numOfLayers-1,1);
trained_b = cell(numOfLayers-1,1);

pretrain_input = cell(numOfImages,numOfLayers-1);

for i=1:numOfImages
    pretrain_input{i,1} = images(:,i);
end

% create NNs for pre-training
for i=1:numOfLayers-1
    tmp_Layers = Layers;
    tmp_Layers{1}.inputNum = Layers{i}.inputNum;
    tmp_Layers{1}.outputNum = Layers{i}.outputNum;
    tmp_Layers{2}.inputNum = Layers{i+1}.inputNum;
    tmp_Layers{2}.outputNum = Layers{i}.inputNum;
    tmp_Layers{3}.inputNum = Layers{i}.inputNum;
    tmp_Layers{3}.outputNum = Layers{i}.outputNum;
    tmp_NN = initialNN(3, tmp_Layers, epsilon);
    
    for j=1:pretrain_iter
            disp(['========== pre-training iteration ' num2str(j) ' =========='])
            for k=1:preTrain_batchSize:numOfImages
                cost = 0;
    %            disp(['========== batch ' num2str(k) ' =========='])
                lo_head = zeros(tmp_Layers{2}.inputNum,1);

                % first of all, forward batchSize times to calc lo
                for l=k:k+preTrain_batchSize-1
                    tmp_NN(1).a = pretrain_input{l,i};
                    tmp_NN(1).z = pretrain_input{l,i};
                    tmp_NN = forward_prop(tmp_NN, 3, activateType);
                    lo_head = lo_head + (1/preTrain_batchSize)*tmp_NN(2).a;
                end
                mean(lo_head)
                for l=k:k+preTrain_batchSize-1
                    tmp_NN(1).a = pretrain_input{l,i};
                    tmp_NN(1).z = pretrain_input{l,i};
                    tmp_NN = forward_prop(tmp_NN, 3, activateType);
     %               cost = cost +  sum( (tmp_NN(numOfLayers).a - tmp_NN(1).a).^2 );
                    tmp_NN = pre_train_back_prop(tmp_NN, numOfLayers, pretrain_input{l,i}, activateType, lo_head, lo, beta);                
                    pretrain_input{l,i+1} = tmp_NN(2).a;
                    
                end
      %          cost
                tmp_NN = updateParas(tmp_NN, learning_rate, preTrain_batchSize, weight_decay, momentum);
            end
            learning_rate = learning_rate * learning_rate_decay;
    end
    NN(i).w = tmp_NN(1).w;
    NN(i).b = tmp_NN(1).b;
end

disp('========== pre-training finished ==========')
disp(['final sparsity = ' num2str(mean(lo_head))])

% end of pre-training
learning_rate = data{1}.learning_rate;

for i=1:max_iter
    disp(['==========iteration ' num2str(i) '=========='])
    perm = randperm(numOfImages);
    images = images(:,perm);
    labels = labels(perm);
    for j=1:numOfImages
            %        disp(['==========process image ' num2str(j) '=========='])
        % put images(training sample) in first layer
        NN(1).a = images(:,j);
        % forward propagation
        NN = forward_prop(NN, numOfLayers, activateType);
        % backward propagation
        NN = back_prop(NN, numOfLayers, labels(j), activateType);
        counter = counter + 1;
        if(mod(counter, batchSize) == 0)        
            NN = updateParas(NN, learning_rate, counter, weight_decay, momentum);
            counter = 0;
        end
    end
    iter_test(testImages, testLabels, test_numOfImages, NN, activateType);
    learning_rate = learning_rate * learning_rate_decay;
end
