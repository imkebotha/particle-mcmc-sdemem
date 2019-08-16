function r = normt(mu, Sig, u)
    % Transforms u ~ N(0, 1) to r ~ N(mu, Sig). 
    
    k = length(mu);
    Lcf = chol(Sig, 'lower');
    mu = reshape(mu, 1, k);
    r = mu + u*Lcf;
end