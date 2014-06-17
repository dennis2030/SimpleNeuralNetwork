function outputM = calcActivate(inputM,activateType)
    % sigmoid
    if activateType == 1
        outputM = 1./(1.+exp(-inputM));
    % tanh
    else
        tmpM = tanh(inputM);
        outputM = (tmpM +1) .*0.45;
    end
end
