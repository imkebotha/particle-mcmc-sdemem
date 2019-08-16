function [chain, X0, lB, acct, accr, ct] = ...
    cCWPM(iterations, which_data, PF)
% Runs the correlated component-wise pseudo marginal method (cCWPM).   
% Displays the acceptance rates and multivariate ESS of the resulting 
% chain.
%
% Input:
%   -   iterations	: number of MCMC iterations
%   -   which_data  : the dataset to use, "real" or "sim1024"
%   -   PF          : propagation function to use in particle filter
%                     EM = 1; MDB = 2; RB = 3;
%
% Output:
%   -   chain       : MCMC chain for theta
%   -   X0          : MCMC chain for X0 random effects
%   -   lB          : MCMC chain for log(beta) random effects
%   -   accr        : acceptance rates for theta parameters
%   -   accr        : acceptance rates for random effects
%   -   ct          : computation time in seconds 
%

%% Data and Tuning Parameters

% constants
GAMMA = 1; SIGMA = 2; RHO = 3; MU_X0 = 4; SIGMA_X0 = 5; MU_BETA = 6; SIGMA_BETA = 7; 

% import data
load(strcat(which_data, '.mat'), 'ds');

% parameter blocks
B1 = GAMMA:RHO;
B2 = MU_X0:SIGMA_BETA;

[Sig, SD_X0, SD_lB, N, D, eps] = tuning_parameters(which_data, PF, true);

% length, starting index and ending index of random numbers for each subject
leni = (ds.T-1)*N*(D+1); 
E = cumsum(leni);
S = [1 ; (E(1:end-1) + 1)];
ind = 1:ds.M; 

% inline functions
loglike = @(theta, x0, lb, u) cGPF(theta, N, x0, lb, D, PF, ds, u);
updateU = @(U, mu) updateRN(leni, 0, ds.M, false, mu, U);

%% MCMC
tic

% preallocate
tchain = zeros(iterations, 7);  % transformed chain
X0 = zeros(iterations, ds.M);      % random effects - X0
lB = zeros(iterations, ds.M);      % random effects - betas

acc_theta = zeros(1, 2);        % acceptance rates (tchain)
acc_re = zeros(1, ds.M);           % acceptance rates (random effects)

% initial values
[tchain(1, :), X0(1, :), lB(1, :)] = initial_values(which_data);
theta = tchain(1, :); x0 = X0(1, :); lb = lB(1, :);

U = updateRN(leni, 0, ds.M, false);    % random numbers
ll = loglike(theta, x0, lb, U);     % log-likelihood

mu = 1; % first block/subject (block pseudo-marginal)

for i = 1:(iterations-1)
    % display progress
    if mod(i, round(iterations/10)) == 0
        fprintf("Running... %i%% complete\n", round(i/iterations*100));
    end

    %% Random Effects: X0 and beta

    % propose new random numbers
    new_U = updateU(U, mu);

    % propose new x0
    new_X0 = normrnd(x0, SD_X0);
    new_lB = normrnd(lb, SD_lB);

    % calculate posteriors
    log_posterior = ll + logeta(x0, lb, theta);

    ll_new = loglike(theta, new_X0, new_lB, new_U); 
    log_posterior_new = ll_new + logeta(new_X0, new_lB, theta);

    % Metropolis-Hastings ratio
    MHRatio = exp(log_posterior_new - log_posterior);

    % accept/reject
    I = rand(1, ds.M) < MHRatio;

    % update values
    x0(I) = new_X0(I);
    lb(I) = new_lB(I);
    ll(I) = ll_new(I);
    
    % update acceptance rates
    acc_re = acc_re + I;

    % update random numbers
    for m = ind(I)
        U(S(m):E(m)) = new_U(S(m):E(m));
    end

    %% hyperparameters: mu_X0, sigma_X0, mu_beta, sigma_beta

    % propose new values
    proposal = theta;
    [proposal(B2), q, qs] = MALA_phi_eta(theta(B2), x0, lb, Sig(B2, B2), eps);

    % calculate posteriors
    log_posterior = sum(logeta(x0, lb, theta)) + logprior(theta);
    log_posterior_new = sum(logeta(x0, lb, proposal)) + logprior(proposal);

    % Metropolis-Hastings ratio
    MHRatio = exp(log_posterior_new - log_posterior + log(qs) - log(q)); 

    % accept/reject
    if (rand < MHRatio)
        theta(B2) = proposal(B2);
        acc_theta(2) = acc_theta(2) + 1;
    end

    %% Fixed Model Parameters: gamma, sigma, rho

    % propose new random numbers
    new_U = updateU(U, mu);

    % propose new values
    proposal = theta;
    proposal(B1) = mvnrnd(tchain(i, B1), Sig(B1, B1));

    % calculate posteriors
    log_posterior = sum(ll) + logprior(theta);

    ll_new = loglike(proposal, x0, lb, new_U); 
    log_posterior_new = sum(ll_new) + logprior(proposal);

    % Metropolis-Hastings ratio
    MHRatio = exp(log_posterior_new - log_posterior); 

    % accept/reject
    if (rand < MHRatio) 
        theta(B1) = proposal(B1);
        ll = ll_new;
        acc_theta(1) = acc_theta(1) + 1;
        U = new_U;
    end


    %% update chains

    mu = mod(mu, ds.M) + 1;
    tchain(i+1, :) = theta;
    X0(i+1, :) = x0;
    lB(i+1, :) = lb;

end

%% Results

% computation time
ct = toc;

% transform chain back
chain = exp(tchain);
chain(:, [RHO MU_X0 MU_BETA]) = tchain(:, [RHO MU_X0 MU_BETA]);

% acceptance rates
acct = acc_theta/iterations;
accr = acc_re/iterations;
array2table(acct, 'VariableNames', {'AR_gamma_sigma_rho', 'AR_hyperparameters'})

% effective sample size
ESS = [multiESS(chain), multiESS(chain(:, GAMMA:RHO)), ...
    multiESS(chain(:, MU_X0:SIGMA_X0)), ...
    multiESS(chain(:, MU_BETA:SIGMA_BETA))];
array2table(ESS, 'VariableNames', ...
    {'mESS_theta', 'mESS_gamma_sigma_rho' ,'mESS_X0', 'mESS_BETA'})
 
end