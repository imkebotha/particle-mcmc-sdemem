function f = Laplace_MDB(eta, theta, m, ds)
% Returns -P(Y_m|X*_m, theta)P(eta_m|theta), where X* is an approximation 
% of the latent states using the mean of the MDB.
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
log_g = theta(GAMMA);                       % log gamma
G = exp(2*log_g - log(2));                  % gamma^2/2
s_sq = exp(2*theta(SIGMA));                 % sigma^2
rho = theta(RHO);                           % rho
x0 = eta(1);                                % X0m
lb = eta(2); beta = exp(lb);                % betam

% initialise
xs = zeros(ds.T(m),1);
xs(1) = x0;

i = 2;
for t = (ds.Ti(m)+1):(ds.Ti(m)+ds.T(m)-1)
    dt = ds.times(t) - ds.times(t-1);
    dk = dt;
    
    ak = beta + G*(1 - exp(2*(rho-1)*xs(i-1))); % drift term
    bk = exp(2*(log_g + (rho-1)*xs(i-1))); % diffusion term squared
        
    K = (bk*dk + s_sq);
    mu_MDB = (ak*s_sq + bk.*(ds.Y(t) - xs(i-1)))./K;
    
    xs(i) = xs(i-1) + dt*mu_MDB; 
    i = i + 1;
end

% calculate weights -P(Y_m|X*_m, theta)P(eta_m|theta)
f = -(sum(lognorm(ds.Y((ds.Ti(m)):(ds.Ti(m)+ds.T(m)-1)), xs, exp(theta(SIGMA)))) ...
            + lognorm(x0, theta(MU_X0), exp(theta(SIGMA_X0)))...
            + lognorm(lb, theta(MU_BETA), exp(theta(SIGMA_BETA))));

if  ~isfinite(f) || ~isreal(f) || f > 1e100
    f = 1e100;
end

end