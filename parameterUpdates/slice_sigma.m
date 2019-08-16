function s = slice_sigma(s, sfun)
% Sample sigma using a slice sampler
%
% Input:
%   -   s       : current value of sigma
%   -   sfun    : log of the full (unnormalised) conditional posterior of
%                 sigma

lower = 0; % set lower limit to 0 for simplicity
upper = s;
    
slice = sfun(upper) - exprnd(1);

% step out
while (sfun(upper) > slice)
    upper = upper + 0.1;
end

while(1)
    sprop = unifrnd(lower,upper);
    if (sfun(sprop) > slice)
        s = sprop;
        break;
    end
    
    % shrink
    if (sprop < s)
        lower = sprop;
    else
        upper = sprop;
    end
end
    
end

