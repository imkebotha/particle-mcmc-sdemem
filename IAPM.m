function [chain, ar, ct] = ... 
    IAPM(iterations, which_data, PF, ID)
% Runs the individual-augmentation pseudo-marginal method (IAPM).  
% Displays the acceptance rates and multivariate ESS of the resulting 
% chain.
%
% Input:
%   -   iterations	: number of MCMC iterations
%   -   which_data  : the dataset to use, "real" or "sim1024"
%   -   PF          : propagation function to use in particle filter
%                     EM = 1; MDB = 2; RB = 3;
%   -   ID          : importance density to use for importance sampler
%                     PRIOR = 1, L-ODE = 2, LAPLACE-MDB = 3
%
% Output:
%   -   chain       : MCMC chain for theta
%   -   ar          : acceptance rate 
%   -   ct          : computation time in seconds 
%

%% Data and Tuning Parameters

% constants
GAMMA = 1; SIGMA = 2; RHO = 3; MU_X0 = 4; SIGMA_X0 = 5; MU_BETA = 6; SIGMA_BETA = 7;

% import data
load(strcat(which_data, '.mat'), 'ds');
[Sig, ~, ~, N, D] = tuning_parameters(which_data, PF, false, ID); L = N;

loglike = @(theta) IAPM_loglike(theta, N, D, L, ID, PF, ds);

%% MCMC

%% initialise

tic

% preallocate
tchain = zeros(iterations, 7); % transformed chain

% initial values
acc = 0; 
tchain(1, :) = initial_values(which_data);
log_posterior = loglike(tchain(1, :)) + logprior(tchain(1, :));

for i = 1:(iterations-1) % i = i + 1
    % display progress
    if mod(i, round(iterations/10)) == 0
        fprintf("Running... %i%% complete\n", round(i/iterations*100));
    end

    %% Update theta

    % propose new theta
    proposal = mvnrnd(tchain(i, :), Sig);

    % calculate posterior
    log_posterior_new = loglike(proposal) + logprior(proposal);

    % Metropolis-Hastings ratio
    MHRatio = exp(log_posterior_new - log_posterior);

    % accept/reject 
    if (rand < MHRatio) 
        tchain(i+1, :) = proposal;
        log_posterior = log_posterior_new;
        acc = acc + 1;
    else
        tchain(i+1, :) = tchain(i, :);
    end

end
%% results

% computation time
ct = toc;

% transform chain back
chain = exp(tchain);
chain(:, [RHO MU_X0 MU_BETA]) = tchain(:, [RHO MU_X0 MU_BETA]);

% acceptance rate
ar = acc/iterations;
array2table(ar)

% effective sample size
ESS = [multiESS(chain), multiESS(chain(:, GAMMA:RHO)), ...
    multiESS(chain(:, MU_X0:SIGMA_X0)), ...
    multiESS(chain(:, MU_BETA:SIGMA_BETA))];
array2table(ESS, 'VariableNames', ...
    {'mESS_theta', 'mESS_gamma_sigma_rho' ,'mESS_X0', 'mESS_BETA'})

end