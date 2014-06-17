function NN = back_prop(NN, numOfLayers, label, activateType)

% initialize
deltas = cell(numOfLayers,1);

% process for the last layer
labelV = zeros(length(NN(numOfLayers).a),1);
labelV(label+1) = 1;
A = -(labelV - NN(numOfLayers).a);
B = calc_f_pron(NN(numOfLayers).a , activateType);
deltas{numOfLayers} = A .* B;
for i=numOfLayers-1:-1:1
    deltas{i} = NN(i).w' * deltas{i+1} .* calc_f_pron( NN(i).a, activateType);
    NN(i).delta_w = NN(i).delta_w + deltas{i+1} * NN(i).a';
    NN(i).delta_b = NN(i).delta_b + deltas{i+1};
%    i
%    size(NN(i).delta_b)
end

end
