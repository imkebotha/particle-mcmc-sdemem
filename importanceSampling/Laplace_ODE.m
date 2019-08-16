function f = Laplace_ODE(eta, theta, m, ds)
% Returns -P(Y_m|X*_m, theta)P(eta_m|theta), where X* is an approximation 
% of the latent states using the ODE specified by the drift of the SDEMEM.
%
% Input:
%   -   eta     : value of random effects for subject m
%   -   theta   : current value of theta
%   -   m       : current subject
%   -   ds      : struct containing dataset

% constants
GAMMA = 1; SIGMA = 2; RHO = 3; MU_X0 = 4; SIGMA_X0 = 5; 
MU_BETA = 6; SIGMA_BETA = 7; 

lognorm = @(x, m, s) -0.5*log(2*pi*s^2)-0.5*(1/s^2)*(x - m).^2;

% parameters
x0 = eta(1);                    % X0m
lb = eta(2); beta = exp(lb);    % betam

% drift ODE
indm = (ds.Ti(m)):(ds.Ti(m)+ds.T(m)-1);
t = ds.times(indm);

if (theta(RHO) == 1)
    xs = x0 + beta*t;
else
    g = 0.5*exp(2*theta(GAMMA));    % 0.5*gamma^2
    a = (beta + g);                 % beta + 0.5*gamma^2
    b = 2*(theta(RHO)-1);           % 2(rho - 1)
    xs = log(a./(g + exp(-a*b*t).*(a*exp(-b*x0) - g)))/b;
end

% calculate weights -P(Y_m|X*_m, theta)P(eta_m|theta)
f = -(sum(lognorm(ds.Y(indm), xs, exp(theta(SIGMA)))) ...
            + lognorm(x0, theta(MU_X0), exp(theta(SIGMA_X0)))...
            + lognorm(lb, theta(MU_BETA), exp(theta(SIGMA_BETA))));
        
if  ~isfinite(f) || ~isreal(f) || f > 1e100
    f = 1e100;
end

end