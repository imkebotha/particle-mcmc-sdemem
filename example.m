% Script which shows how to run each of the methods. Each section shows a
% different method and plots the resulting chain. 
%
% Method inputs:
%   -   iterations	: number of MCMC iterations
%   -   which_data  : the dataset to use, "real" or "sim1024"
%   -   PF          : propagation function to use in particle filter
%                     EM = 1; MDB = 2; RB = 3;
%   -   ID          : importance density to use for importance sampler
%                     PRIOR = 1, L-ODE = 2, LAPLACE-MDB = 3

% add folders to path if necessary
% addpath(genpath('.'));

parfor (i = 1:2), end % start parallel pool

%% IAPM

iterations = 500;
which_data = "real";
PF = 2;
ID = 3;

[chain, ar, ct] = IAPM(iterations, which_data, PF, ID);

plot_chain(chain);

%% cIAPM

iterations = 500;
which_data = "real";
PF = 2;
ID = 3;

[chain, ar, ct] = cIAPM(iterations, which_data, PF, ID);

plot_chain(chain);

%% CWPM

iterations = 500;
which_data = "real";
PF = 2; 

[chain, X0, lB, acct, accr, ct] = CWPM(iterations, which_data, PF);

plot_chain(chain);

%% cCWPM

iterations = 500;
which_data = "real";
PF = 2; 

[chain, X0, lB, acct, accr, ct] = cCWPM(iterations, which_data, PF);

plot_chain(chain);

%% MPM

iterations = 500;
which_data = "real";
PF = 2; 

[chain, X0, lB, acct, accr, ct] = cMPM(iterations, which_data, PF);

plot_chain(chain)

