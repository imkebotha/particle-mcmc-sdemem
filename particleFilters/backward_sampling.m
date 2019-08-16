function [Xk, Bk, Xkf] = backward_sampling(D, Xn, logW, logNW, Tm, theta, lB)%
    % Returns latent states and associated ancestral lineage for a single 
    % individual using the backward sampling method of Lindsten and Schon 
    % (2012)
    %
    % Input:
    %   -   D       : level of discretisation (number of intermediate
    %                 points)
    %   -   Xn      : set of latent states
    %   -   logW    : log of weights of Xn
    %   -   logNW   : log of normalised weights for Xn at time Tm
    %   -   Tm      :  measurement times for subject m
    %   -   theta   : current value of theta
    %   -   lB      : log(beta) value for subject m
    %
    % Output:
    %   -   Xk  : new invariant path (without intermediate times)
    %   -   Bk  : ancestral lineage of Xk
    %   -   Xkf : new invariant path (with intermediate times)
    
    GAMMA = 1; RHO = 3;

    % initialise
    len = length(Tm);
    Bk = zeros(len, 1);
    Xk = zeros(len, 1);
    Xkf = zeros(size(Xn, 2), 1);
    N = length(logNW);
   
    % model parameters
    beta = exp(lB);
    rho = theta(RHO); 
    G = exp(2*theta(GAMMA) - log(2)); % 0.5*gamma^2
    
    % transition density (EM approximation)
    lognorm = @(x, m, s) -0.5*log(2*pi*s.^2)-0.5*(1./s.^2).*(x - m).^2;
    logf = @(x, xg, dt) lognorm(x, ...
        xg + (beta + G*(1 - exp(2*(rho-1)*xg)))*dt, ...     % mean
        exp(theta(GAMMA) + (rho-1)*xg + 0.5*log(dt)));      % standard deviation
    
    % draw single value
    k_new = randsample(1:N, 1, true, exp(logNW)); 
    
    % sample ancestral lineage
    Bk(end) = k_new;
    Xk(end) = Xn(Bk(end), end);
    Xkf(end) = Xn(Bk(end), end);
    
    n = size(Xn, 2)-1;
    i = len; 
    for j = n:-1:1
        
        if (mod(j, D) == 0)
            i = i - 1;
            dt = (Tm(i+1) - Tm(i))/D;
        end
        
        X = Xkf(j+1); Xp = Xn(:, j);            % current and previous X
        nlogW = logW(:, j) + logf(X, Xp, dt);   % calculate log weights
        nlogNW = nlogW - logsumexp(nlogW);      % normalise weights
        
        % sample ancestor for current X
        bk = randsample(1:N, 1, true, exp(nlogNW)); 
        Xkf(j) = Xn(bk, j);
        
        if (mod(j-1, D) == 0)
            Bk(i) = bk; 
            Xk(i) = Xkf(j);
        end
        
    end
    
    % remove X0
    Bk = Bk(2:end);
    Xk = Xk(2:end);
    Xkf = Xkf(2:end);
end
