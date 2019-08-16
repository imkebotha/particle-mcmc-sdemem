function LL = IAPM_loglike(theta, N, D, L, ID, PF, ds, varargin)
% Returns the log-likelihood estimate P(Y|theta) 
% 
% Input:
%   -   theta   : current value of theta
%   -   N       : number of particles
%   -   D       : level of discretisation 
%   -   L       : number of random effects draws for importance sampler
%   -   ID      : importance density to use for importance sampler
%                 PRIOR = 1, L-ODE = 2, LAPLACE-MDB = 3
%   -   PF      : propagation function to use in particle filter
%                 EM = 1; MDB = 2; RB = 3;
%   -   ds      : struct containing dataset
%   -   varargin : U, vector of random numbers (auxiliary variables)
    
    % preallocate
    LL = zeros(L, ds.M);
    
    if(isempty(varargin))
        %% uncorrelated
        [X0, lB, logREW] = draw_re(theta, L, ID, ds);
        parfor i = 1:L
            LL(i, :) = GPF(theta, N, X0(i, :), lB(i, :), D, PF, ds);
        end
    else
        %% correlated
        U = varargin{:};
        u_re = U(1:2*L*ds.M);
        u_pf = U(2*L*ds.M+1:end);
        
        [X0, lB, logREW] = draw_re(theta, L, ID, ds, u_re);
        parfor i = 1:L 
            LL(i, :) = cGPF(theta, N, X0(i, :), lB(i, :), D, PF, ds, u_pf);
        end
    end
    
    LL = sum(logsumexp(LL + logREW) - log(L));

end