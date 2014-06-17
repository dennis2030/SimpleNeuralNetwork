function NN = pre_train_back_prop(NN, numOfLayers, label, activateType, lo_head, lo, beta)

% initialize
deltas = cell(numOfLayers, 1);


% process for the last layer
A = -(label - NN(numOfLayers).a);
B = calc_f_pron(NN(numOfLayers).a , activateType);
deltas{numOfLayers} = A .* B;

part1 =  (-lo) ./ lo_head;
part2 = (1-lo) ./ (1 - lo_head);
KL_term = part1 + part2;
%KL_term =  (-(lo ./ lo_head))' + ( (1-lo) / ( 1 - lo_head) ) ;
deltas{2} = (NN(2).w' * deltas{3} + beta * KL_term) .* calc_f_pron(NN(2).a ,activateType);
NN(2).delta_w = NN(2).delta_w + deltas{3}*NN(2).a';
NN(2).delta_b = NN(2).delta_b + deltas{3};

deltas{1} = NN(1).w' * deltas{2} .* calc_f_pron( NN(1).a , activateType);
NN(1).delta_w = NN(1).delta_w + deltas{2}*NN(1).a';
NN(1).delta_b = NN(1).delta_b + deltas{2};


end



