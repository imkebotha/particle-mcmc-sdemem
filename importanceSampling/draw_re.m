function [X0, lB, logW] = draw_re(theta, L, ID, ds, varargin)
    % Returns M x L random effects (X0 and log(beta)) draws from the 
    % specified importance density. 
    %
    % Input:
    %   -   theta       : current value of theta
    %   -   L           : number of random effects draws per subject
    %   -   ID          : importance density to use for importance sampler
    %                     PRIOR = 1, L-ODE = 2, LAPLACE-MDB = 3
    %   -   ds          : struct containing dataset
    %   -   varargin    : U, vector of random numbers (auxiliary variables)
    %
    % Output:
    %   -   X0      : draws for X0
    %   -   lB      : draws for log(beta)
    %   -   logW    : ratio prior/importance density for each draw
    
    % constants
    MU_X0 = 4; SIGMA_X0 = 5; MU_BETA = 6; SIGMA_BETA = 7; 
    PRIOR = 1; ODE = 2; MDB = 3;

    lognormpdf = @(x, m, s) -0.5*log(2*pi*s^2)-0.5*(1/s^2)*(x - m).^2;

    % parameters
    M = ds.M;
    mv0 = theta(MU_X0); sv0 = exp(theta(SIGMA_X0));
    mb = theta(MU_BETA); sb = exp(theta(SIGMA_BETA));

    % draw random numbers if none have been supplied
    if isempty(varargin)    
        u = nrqmc(L*M, 2);
    else
        u = varargin{:};
        u = reshape(u, L*M, 2);
    end

    % initialise  
    logW = zeros(L, M);

    if (ID == PRIOR)
        X0 = reshape(normt(mv0, sv0^2, u(:, 1)), L, M);
        lB = reshape(normt(mb, sb^2, u(:, 2)), L, M);
    else
        % initialise
        X0 = zeros(L, M);
        lB = zeros(L, M);
        ini = [3 0];

        % set fminunc options
        warning('off','all')
        options = optimoptions('fminunc', 'Display', 'off');

        % set function to minimize
        if (ID == ODE)
            f = @(eta, m) Laplace_ODE(eta, theta, m, ds);
        else
            f = @(eta, m) Laplace_MDB(eta, theta, m, ds);
        end

        % default covariance - 0.5 x diagonal matrix of prior variance
        Sig_def = 0.5*diag([sv0^2 sb^2]);

        for i = 1:M            
            % calculate mode and hessian
            [mu_re, ~, ~, ~, ~, hessian] = fminunc(@(eta) f(eta, i), ini, options);

            % covariance matrix
            Sigma_re = Sig_def;
            if ID == MDB
                Sigma_hess = inv(hessian/2);
                [~,p] = chol(Sigma_hess);

                if p == 0
                    Sigma_re = Sigma_hess;
                end
            end

            % draw random effects
            ind = (i-1)*L+1;  
            ui = u(ind:(ind+L-1), :);

            r_rand = normt(mu_re, Sigma_re, ui);
            X0(:, i) = r_rand(:, 1);
            lB(:, i) = r_rand(:, 2);

            % calculate weights
            logW(:, i) = lognormpdf(X0(:, i), mv0, sv0) ...
                + lognormpdf(lB(:, i), mb, sb) ...
                - logmvnpdf(r_rand, mu_re, Sigma_re);
        end

        warning('on','all');
    end
end