function p = logprior(theta)
    % Evaluates the log(prior) for the (transformed) theta parameters

    GAMMA = 1; SIGMA = 2; RHO = 3; MU_X0 = 4; SIGMA_X0 = 5; MU_BETA = 6; SIGMA_BETA = 7; 

    lognorm = @(x, m, log_s) ...
        -0.5*log(2*pi) - log_s - 0.5*(x - m).^2/exp(2*log_s); 
    
    logHN = @(logY, s) 0.5*log(2/pi)-log(s)-exp(2*logY)/(2*s^2) + logY;

    p = ...
        logHN(theta(GAMMA), 5) + ... 
        logHN(theta(SIGMA), 5) + ... 
        lognorm(theta(RHO), 1, log(0.5)) + ... 
        lognorm(theta(MU_X0), 3, log(4)) + ... 
        logHN(theta(SIGMA_X0), 5) + ... 
        lognorm(theta(MU_BETA), 0, log(4)) + ... 
        logHN(theta(SIGMA_BETA), 5);

end