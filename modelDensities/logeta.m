function p = logeta(X0, lB, theta)
    % Evaluates the density log(P(eta|theta)), where eta contains the 
    % (transformed) random effects 
    
    MU_X0 = 4; SIGMA_X0 = 5; MU_BETA = 6; SIGMA_BETA = 7; 
    
    lognorm = @(x, m, log_s) ...
        -0.5*log(2*pi) - log_s - 0.5*(x - m).^2/exp(2*log_s); 

    p = lognorm(X0, theta(MU_X0), theta(SIGMA_X0)) + ...
    lognorm(lB, theta(MU_BETA), theta(SIGMA_BETA));
end
