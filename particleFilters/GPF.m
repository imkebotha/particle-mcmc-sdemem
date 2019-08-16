function LL = GPF(theta, N, X0, lB, D, PF, ds)
    % Returns a particle filter estimate of P(Y|eta, theta), where eta are
    % the random effects
    %
    % Input:
    %   -   theta   : current value of theta
    %   -   N       : number of particles
    %   -   X0      : X0 random effects for all subjects
    %   -   lB      : log(beta) random effects for all subjects
    %   -   D       : level of discretisation 
    %   -   PF      : propagation function to use in particle filter
    %                 EM = 1; MDB = 2; RB = 3;
    %   -   ds      : struct containing dataset
    
    % initialise
    SIGMA = 2; s = exp(theta(SIGMA)); 
    lognorm = @(x, m) -0.5*log(2*pi*s^2)-0.5*(1/s^2)*(x - m).^2;
    
    Y = ds.Y; T = ds.T; Ti = ds.Ti; M = ds.M; times = ds.times;
    
    LL = zeros(1, M);

    % only parallelize if average number of observations per subject is
    % greater than 10
    if (mean(T) > 10)
        
        % get rid of broadcast variables
        Yc = arrayfun(@(a, b) Y(a:b), Ti, Ti + T - 1, 'UniformOutput', false);
        timesc = arrayfun(@(a, b) times(a:b), Ti, Ti + T - 1, 'UniformOutput', false);
        
        parfor m = 1:M   
            Ym = Yc{m};
            timesm = timesc{m};
            
            % at t = 1
            Xn = repmat(X0(m), N, 1);
            logW = lognorm(Ym(1), Xn) - log(N);
            logNW = logW - logsumexp(logW);
            LL(m) = LL(m) + logsumexp(logW);

            for t = 2:T(m)

                % adaptive resampling
                ESS = exp(-logsumexp(2*logNW));
                if (ESS < N/2)
                    Xn = randsample(Xn, N, true, exp(logNW));
                    logNW = repmat(-log(N), N, 1);
                end

                % simulate particles forward
                time = timesm(t) - timesm(t-1);
                [Xn, logw] = simulate_forward(Ym(t), Xn, theta, lB(m), D, N, time, PF); 

                % log weights 
                logW = lognorm(Ym(t), Xn) + logw + logNW;

                % normalize weights 
                logNW = logW - logsumexp(logW);

                % log likelihood
                LL(m) = LL(m) + logsumexp(logW);
            end

        end
    else
        
        for m = 1:M   

            % at t = 1
            Xn = repmat(X0(m), N, 1);
            logW = lognorm(Y(Ti(m)), Xn) - log(N);
            logNW = logW - logsumexp(logW);
            LL(m) = LL(m) + logsumexp(logW);

            for t = (Ti(m)+1):(Ti(m)+T(m)-1)

                % adaptive resampling
                ESS = exp(-logsumexp(2*logNW));
                if (ESS < N/2)
                    Xn = randsample(Xn, N, true, exp(logNW));
                    logNW = repmat(-log(N), N, 1);
                end

                % simulate particles forward 
                time = times(t) - times(t-1);
                [Xn, logw] = simulate_forward(Y(t), Xn, theta, lB(m), D, N, time, PF); 

                % log weights 
                logW = lognorm(Y(t), Xn) + logw + logNW;

                % normalize weights 
                logNW = logW - logsumexp(logW);

                % log likelihood
                LL(m) = LL(m) + logsumexp(logW);
            end

        end
    end
end