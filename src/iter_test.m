function iter_test(images, labels, numOfImages, NN, activateType)

right_guess = 0;
guess = zeros(10,1);
numOfLayers = length(NN);
disp('======testing======')
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

end
