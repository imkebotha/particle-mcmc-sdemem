function Z = nrqmc(m, k)
    % Returns an m x k matrix of standard normal random numbers generated
    % using RQMC.
    
    P = sobolset(k);                         % Get Sobol sequence of dim k
    P = scramble(P,'MatousekAffineOwen');    % Scramble Sobol points;
    Z = net(P, m);                           % Get first m numbers

    % convert to standard normal numbers
    for i = 1:k
        Z(:, i) = norminv(Z(:, i), 0, 1);
    end

end