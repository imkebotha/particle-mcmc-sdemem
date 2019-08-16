function [Xk_new, Bk_new, Xf_new] = CSMC(Xkf, Bk, X0, lB, theta, D, N, PF, ds) 
    % Returns latent states and associated ancestral lineage for a single 
    % individual using a conditional particle filter and the backward 
    % sampling method of Lindsten and Schon (2012)
    %
    % Input:
    %   -   Xkf     : invariant path (with intermediate times)
    %   -   Bk      : ancestral lineage of Xk
    %   -   X0      : X0 random effects for all subjects
    %   -   lB      : log(beta) random effects for all subjects
    %   -   theta   : current value of theta
    %   -   D       : level of discretisation 
    %   -   N       : number of particles
    %   -   simf    : function used to simulate particles forward
    %   -   ds      : struct containing dataset
    %
    % Output:
    %   -   Xk_new  : new invariant path (without intermediate times)
    %   -   Bk_new  : ancestral lineage of Xk
    %   -   Xkf_new : new invariant path (with intermediate times)
    
    % observation density
    log_s = theta(2); % log(sigma)
    lognorm = @(x, m) -0.5*log(2*pi) - log_s - 0.5*(x - m).^2./exp(2*log_s); 
    
    Y = ds.Y; T = ds.T; Ti = ds.Ti; M = ds.M; times = ds.times;
    
    % get rid of broadcast variables for parfor
    Yc = arrayfun(@(a, b) Y(a:b), Ti, Ti + T - 1, 'UniformOutput', false);
    timesc = arrayfun(@(a, b) times(a:b), Ti, Ti + T - 1, 'UniformOutput', false);
    Bkc = arrayfun(@(a, b) Bk(a:b), Ti - (1:M)' + 1, Ti + T - (1:M)' - 1, 'UniformOutput', false);
    Xkfc = arrayfun(@(a, b) Xkf(a:b), (Ti - (1:M)')*D + 1, (Ti - (1:M)')*D + (T-1)*D, 'UniformOutput', false);
    
    % initialise
    Xkc_new = cell(M, 1);
    Bkc_new = cell(M, 1); 
    Xfc_new = cell(M, 1);
      
    % only parallelise if average number of observations per subject is
    % greater than 10
    if (mean(T) > 10)
        
        parfor m = 1:M 
            Ym = Yc{m};
            timesm = timesc{m};
            Bkm = Bkc{m};
            Xkfm = Xkfc{m};

            % initialise
            Xn = zeros(N, (T(m)-1)*D+1);
            logW = zeros(N, (T(m)-1)*D+1);

            % at t = 1
            Xn(:, 1) = repmat(X0(m), N, 1); 
            logW(:, 1) = lognorm(Ym(1), Xn(:, 1)) - log(N);
            logNW = logW(:, 1) - logsumexp(logW(:, 1));

            for i = 2:T(m)

                % calculate indices
                r1 = (i-2)*D + 2; r2 = r1 + D - 1; % Xn

                b = Bkm(i-1); % invariant path at time t
                not_b = [1:b-1 b+1:N];

                % resampling step 
                I = randsample(1:N, N-1, true, exp(logNW));
                Xn(not_b, r1) = Xn(I, r1-1);
                Xn(b, r1) = Xkfm(r1-1);

                % simulate particles forward
                time =  timesm(i) - timesm(i-1); 
                [Xn(:, r1:r2), logW(:, r1:r2)] = ...
                    sim_forward_mpm(Ym(i), Xn(:, r1), Xkfm((r1:r2)-1), b, not_b, theta, lB(m), D, N, time, PF);

                % log weights 
                logW(:, r2) = logW(:, r2) + lognorm(Ym(i), Xn(:, r2)) - log(N); 

                % normalize weights 
                logNW = logW(:, r2) - logsumexp(logW(:, r2));
            end

            %% draw new invariant path
            
            [Xkc_new{m}, Bkc_new{m}, Xfc_new{m}] = ...
                backward_sampling(D, Xn, logW, logNW, timesm, theta, lB(m)); 
            
        end

    else
        
        for m = 1:M 
            Ym = Yc{m};
            timesm = timesc{m};
            Bkm = Bkc{m};
            Xkfm = Xkfc{m};

            % initialise
            Xn = zeros(N, (T(m)-1)*D+1);
            logW = zeros(N, (T(m)-1)*D+1);

            % at t = 1
            Xn(:, 1) = repmat(X0(m), N, 1); 
            logW(:, 1) = lognorm(Ym(1), Xn(:, 1)) - log(N);
            logNW = logW(:, 1) - logsumexp(logW(:, 1));

            for i = 2:T(m) 

                % calculate indices
                r1 = (i-2)*D + 2; r2 = r1 + D - 1; % Xn

                b = Bkm(i-1); % invariant path at time t
                not_b = [1:b-1 b+1:N];

                % resampling step 
                I = randsample(1:N, N-1, true, exp(logNW));
                Xn(not_b, r1) = Xn(I, r1-1);
                Xn(b, r1) = Xkfm(r1-1);

                % simulate particles forward
                time =  timesm(i) - timesm(i-1); 
                [Xn(:, r1:r2), logW(:, r1:r2)] = ...
                    sim_forward_mpm(Ym(i), Xn(:, r1), Xkfm((r1:r2)-1), b, not_b, theta, lB(m), D, N, time, PF);

                % log weights 
                logW(:, r2) = logW(:, r2) + lognorm(Ym(i), Xn(:, r2)) - log(N); 

                % normalize weights 
                logNW = logW(:, r2) - logsumexp(logW(:, r2));
            end

            %% draw new invariant path

            [Xkc_new{m}, Bkc_new{m}, Xfc_new{m}] = ...
                backward_sampling(D, Xn, logW, logNW, timesm, theta, lB(m)); 
            
        end
    end
    
    % convert cell arrays back to vectors
    Xk_new = cell2mat(Xkc_new);
    Bk_new = cell2mat(Bkc_new);
    Xf_new = cell2mat(Xfc_new);
    
end