function NN = forward_prop(NN, numOfLayers, activateType)
numOfLayers = length(NN);

for i=1:numOfLayers-1
   z = NN(i).w * NN(i).a + NN(i).b;
   NN(i+1).a = calcActivate(z, activateType);
end

end
