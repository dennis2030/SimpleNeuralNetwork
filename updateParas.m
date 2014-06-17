function NN = updateParas(NN, learning_rate, numOfImages, weight_decay, momentum)

numOfLayers = length(NN);

oneOverNumOfImages = 1.0/numOfImages;

for i=1:numOfLayers
    gradient_w = oneOverNumOfImages*NN(i).delta_w + weight_decay * NN(i).w + momentum * NN(i).last_gradient_w;
    gradient_b = oneOverNumOfImages*NN(i).delta_b + momentum * NN(i).last_gradient_b;

    NN(i).last_gradient_w = gradient_w;
    NN(i).last_gradient_b = gradient_b;
    NN(i).w = NN(i).w - learning_rate * gradient_w;
    NN(i).b = NN(i).b - learning_rate * gradient_b;
    
    NN(i).delta_w = 0;
    NN(i).delta_b = 0;
end


