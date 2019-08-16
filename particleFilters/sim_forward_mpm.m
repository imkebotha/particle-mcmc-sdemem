function [X, logW] = sim_forward_mpm(Y, Xp, Xk, b, not_b, theta, lB, D, N, time, PF)
    % Returns particles that have been simulated forward using the
    % specified propagation function
    %
    % Input:
    %   -   Y           : data vector
    %   -   Xp          : current set of particles
    %   -   Xk          : Invariant path
    %   -   b           : location of invariant path at current time point
    %   -   not_b       : location of rest of particles {1, ..., b-1, b+1,
    %                     ..., N}
    %   -   theta       : current value of theta
    %   -   lB          : log(beta) value for subject m
    %   -   D           : level of discretisation 
    %   -   N           : number of particles
    %   -   time        : time between the current and next observation
    %   -   PF          : propagation function to use 
    %                     EM = 1; MDB = 2; RB = 3;
    %
    % Output:
    %   -   X       : moved particles (including intermediate times)
    %   -   logW    : product of P(X|eta, theta)/hat{P}(X|eta, theta) for
    %                 all intermediate time points

    lognorm = @(x, m, s) -0.5*log(2*pi*s.^2)-0.5*(1./s.^2).*(x - m).^2;
    
    % constants
    MDB = 2; RB = 3;
    GAMMA = 1; SIGMA = 2; RHO = 3;
    dt = time/D;
    
    % parameters
    log_g = theta(GAMMA);                   % log gamma
    G = exp(2*log_g - log(2));              % gamma^2/2
    s_sq = exp(2*theta(SIGMA));             % sigma^2
    rho = theta(RHO); 
    beta = exp(lB);
    
    % initialise
    X = zeros(N, D);
    logW = zeros(N, D);
    lw = zeros(N, 1);
    
    % known values of X
    X(:, 1) = Xp;
    X(b, :) = Xk;
    x = Xp; % current x
    
    % residual bridge (RB) - solve drift ODE
    if (PF == RB)
        % solve for eta_t
        tspan = 0:dt:time;
        
        a = (beta + G);
        rt = 2*(rho-1);

        if (theta(RHO) == 1)
            Xd = Xp + beta*tspan;
        else
            Xd = log(a./(G + exp(-a*rt*tspan).*(a*exp(-rt*Xp) - G)))/rt;
        end
        
        if ~isreal(Xd)
            Xd(imag(Xd)~=0) = 0;
        end
    end
    
    for k = 0:D-1
        dk = (time - k*dt);
        
        % drift
        ak = beta + G*(1 - exp(2*(rho-1)*x)); 
        
        % diffusion squared
        lb = 2*(log_g + (rho-1)*x); bk = exp(lb); 
        
        % Euler-Maruyama (EM) bridge
        mu_EM = x + ak*dt; mu = mu_EM;
        sigma_EM = sqrt(bk*dt); sigma = sigma_EM;
        
        % modified diffusion bridge (MDB)
        if (PF == MDB)
            K = (bk*dk + s_sq);
            mu_MDB = (ak*s_sq + bk.*(Y-x))./K;
            psi_MDB = exp(0.5*(lb + log(s_sq + bk*(dk-dt)) - log(K)));

            mu = x + mu_MDB*dt;
            sigma = psi_MDB*sqrt(dt);
            
        % residual bridge (RB)
        elseif (PF == RB)
            r = x - Xd(:, k+1);
            
            dk = (time - k*dt); 
            chord =  (Xd(:, k+2) - Xd(:, k+1))/dt;

            K = (bk*dk + s_sq);
            mu_RB = ak + bk.*(Y - Xd(:, end) - r - (ak-chord)*dk)./K;
            psi_RB = exp(lb + log(s_sq + bk*(dk-dt)) - log(K)); 

            mu = x + mu_RB*dt;
            sigma = sqrt(psi_RB*dt);

        end
         
        % new states
        X(not_b, k+1) = mu(not_b) + sigma(not_b).*randn(N-1, 1);
        x = X(:, k+1);
        
        % weights
        lw = lw + lognorm(x, mu_EM, sigma_EM) - lognorm(x, mu, sigma);
        logW(:, k+1) = lw;
    end
    
    % if the path or weight of a particle is nan, then don't simulate that
    % particle forward
    J = ~isfinite(X + logW);
    if any(J)
        r = logical(min(sum(J, 2), 1));
        X(r, :) = Xp(r);
        logW(r, :) = 0;
        fprintf('.');
    end
end