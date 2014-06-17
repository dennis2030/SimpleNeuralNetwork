function L = initialNN(numOfLayers, Layers, epsilon)

    L(1:numOfLayers) = struct('inputNum', [], 'outputNum', [],'w',[], 'b', [], 'a', [], 'delta_w', 0, 'delta_b', 0, 'last_gradient_w', 0, 'last_gradient_b', 0);
    for i=1:numOfLayers

        % assign random weights acording to (0,epsilon^2) normal distribution
        w = normrnd(0, epsilon, Layers{i}.outputNum, Layers{i}.inputNum);
        b = normrnd(0, epsilon, Layers{i}.outputNum, 1);
        L(i).inputNum = Layers{i}.inputNum;
        L(i).outputNum = Layers{i}.outputNum;
        L(i).w = w;
        L(i).b = b;
    end

end
