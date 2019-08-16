function [X, logW] = simulate_forward(Y, X, theta, lB, D, N, time, PF, varargin)
    % Returns particles that have been simulated forward using the
    % specified propagation function
    %
    % Input:
    %   -   Y           : data vector
    %   -   X           : current set of particles
    %   -   theta       : current value of theta
    %   -   lB          : log(beta) value for subject m
    %   -   D           : level of discretisation 
    %   -   N           : number of particles
    %   -   time        : time between the current and next observation
    %   -   PF          : propagation function to use 
    %                     EM = 1; MDB = 2; RB = 3;
    %   -   varargin    : used to pass U if using correlated methods
    %
    % Output:
    %   -   X       : moved particles
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
    X0 = X;
    
    % initialise
    logW = zeros(N, 1);
    if (isempty(varargin)) 
        BM = randn(N, D); 
    else
        BM = varargin{:};
    end
    
    % residual bridge (RB) - solve drift ODE
    if (PF == RB)
        % solve for eta_t
        tspan = 0:dt:time;
        
        a = (beta + G);
        b = 2*(rho-1);

        if (theta(RHO) == 1)
            Xd = X0 + beta*tspan;
        else
            Xd = log(a./(G + exp(-a*b*tspan).*(a*exp(-b*X0) - G)))/b;
        end
        
        if ~isreal(Xd)
            Xd(imag(Xd)~=0) = 0;
        end
    end
    
    for k = 0:D-1
        dk = (time - k*dt);
        
        % drift
        ak = beta + G*(1 - exp(2*(rho-1)*X)); 
        
        % diffusion squared
        lb = 2*(log_g + (rho-1)*X); bk = exp(lb); 
        
        % Euler-Maruyama (EM) bridge
        mu_EM = X + ak*dt; mu = mu_EM;
        sigma_EM = sqrt(bk*dt); sigma = sigma_EM;
        
        % modified diffusion bridge (MDB)
        if (PF == MDB)
            K = (bk*dk + s_sq);
            mu_MDB = (ak*s_sq + bk.*(Y-X))./K;
            psi_MDB = exp(0.5*(lb + log(s_sq + bk*(dk-dt)) - log(K)));

            mu = X + mu_MDB*dt;
            sigma = psi_MDB*sqrt(dt);
            
        % residual bridge (RB)
        elseif (PF == RB)
            r = X - Xd(:, k+1);
            
            dk = (time - k*dt); 
            chord =  (Xd(:, k+2) - Xd(:, k+1))/dt;

            K = (bk*dk + s_sq);
            mu_RB = ak + bk.*(Y - Xd(:, end) - r - (ak-chord)*dk)./K;
            psi_RB = exp(lb + log(s_sq + bk*(dk-dt)) - log(K)); 

            mu = X + mu_RB*dt;
            sigma = sqrt(psi_RB*dt);

        end
         
        % new states
        X = mu + sigma.*BM(:, k+1);
        
        % weights
        logW = logW + lognorm(X, mu_EM, sigma_EM) - lognorm(X, mu, sigma);
    end
    
    % if the path or weight of a particle is nan, then don't simulate that
    % particle forward
    J = ~isfinite(X + logW);
    if any(J)
        X(J) = X0(J);
        logW(J) = 0;
    end

end