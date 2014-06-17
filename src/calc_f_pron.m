function result = calc_f_pron(a,activateType)

if activateType==1
    result = a .* (1.-a);
else
    result = (sech(a).^2);
end

end
