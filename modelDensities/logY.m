function p = logY(Xk, X0, log_sigma, ds)
% Evaluates the density P(Y|X, sigma^2)
%
% Inputs:
%   -   Xk          : latent states for all individuals m, from t = 2,...,Tm
%   -   X0          : X0 random effects
%   -   log_sigma   : log(sigma)
%   -   ds          : struct containing dataset

% remove Y1 for each individual
Y_red = ds.Y(~ismember(1:length(ds.Y), ds.Ti));

p = -(ds.T*0.5*log(2*pi*exp(2*log_sigma)))' ...
    -(1/(2*exp(2*log_sigma)))*(...
    (ds.Y(ds.Ti)' - X0).^2 + ...
    movesum((Y_red - Xk).^2, ds.T-1));
end