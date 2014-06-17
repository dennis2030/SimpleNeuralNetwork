function pre_train(images, numOfLayers, Layers, epsilon, numOfImages, pretrain_iter, activateType)

    % initialization
    trained_w = cells(numOfLayers-1,1);
    trained_b = cells(numOfLayers-1,1);
   % NNs = cells(numOfLayers-1,1);

    % create NNs for pre-training
    for i=1:numOfLayers-1
        tmp_Layers = Layers;
        tmp_Layers{1}.inputNum = Layers{i}.inputNum;
        tmp_Layers{1}.outputNum = Layers{i}.outputNum;
        tmp_Layers{2}.inputNum = Layers{i+1}.inputNum;
        tmp_Layers{2}.outputNum = Layers{i+1}.outputNum;
        tmp_Layers{3}.inputNum = Layers{i}.inputNum;
        tmp_Layers{3}.outputNum = Layers{i}.outputNum;
        tmp_NN = initialNN(3, tmp_Layers, epsilon);
    %    NNs{i} = tmp_NN;
        for j=1:pretrain_iter
            for k=1:numOfImages
                tmp_NN(1).a = images(:,k);
                tmp_NN = forward_prop(tmp_NN, 3, activateType);
                
            end
        end
    end

    for i=1:numOfImages
    
    end
    

end
