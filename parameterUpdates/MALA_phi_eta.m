function [prop, q, q_star] = MALA_phi_eta(theta, X0, lB, invG, eps)
    % Returns the MALA proposal for the random effects hyperparameters as
    % well as the densities for the proposal ratio q(theta*|theta) and
    % q(theta|theta*)
    %
    % Input:
    %   -   theta   : current value of theta
    %   -   X0      : X0 random effects for all subjects
    %   -   lB      : log(beta) random effects for all subjects
    %   -   invG    : inverse of the pre-conditioning matrix
    %   -   eps     : step-size for MALA
    %
    % Output:
    %   -   prop    : proposed value of theta
    %   -   q       : q(theta*|theta)
    %   -   q_star  : q(theta|theta*)
    
    % constants
    MU_X0 = 1; SIGMA_X0 = 2; MU_BETA = 3; SIGMA_BETA = 4; 
    M = length(X0);
    
    %% propose theta* and calculate q(theta*|theta)
    
    % parameter values (theta)
    mx = theta(MU_X0); sx = theta(SIGMA_X0);
    mb = theta(MU_BETA); sb = theta(SIGMA_BETA);

    sumx = sum(X0); sumx2 = sum((X0-mx).^2); 
    sumb = sum(lB); sumb2 = sum((lB-mb).^2); 

    % gradient (theta)
    gmx = (sumx - M*mx)/exp(2*sx) - (mx-3)/16;
    gsx = 1 - M + sumx2/exp(2*sx) - exp(2*sx)/25;
    gmb = (sumb - M*mb)/exp(2*sb) - mb/16;
    gsb = 1 - M + sumb2/exp(2*sb) - exp(2*sb)/25;

    % q(theta*|theta) 
    mu_star = theta' + eps.^2/2*invG*[gmx gsx gmb gsb]'; mu_star = mu_star';
    cov_star = eps^2*invG;
    prop = mvnrnd(mu_star, cov_star);
    q = mvnpdf(prop, mu_star, cov_star);

    %% calculate q(theta|theta*)
    
    % parameter values (theta*)
    mx = prop(MU_X0); sx = prop(SIGMA_X0);
    mb = prop(MU_BETA); sb = prop(SIGMA_BETA);

    sumx = sum(X0); sumx2 = sum((X0-mx).^2); 
    sumb = sum(lB); sumb2 = sum((lB-mb).^2); 

    % gradient (theta*)
    gmx = (sumx - M*mx)/exp(2*sx) - (mx-3)/16;
    gsx = 1 - M + sumx2/exp(2*sx) - exp(2*sx)/25;
    gmb = (sumb - M*mb)/exp(2*sb) - mb/16;
    gsb = 1 - M + sumb2/exp(2*sb) - exp(2*sb)/25;

    % q(theta | theta*)
    mu_theta = prop' + eps.^2/2*invG*[gmx gsx gmb gsb]'; mu_theta = mu_theta';
    cov_theta = eps^2*invG;
    q_star = mvnpdf(theta, mu_theta, cov_theta);


end