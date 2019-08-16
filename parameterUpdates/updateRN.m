function newU = updateRN(leni, L, M, isIAPM, varargin)
    % Updates the auxiliary variables for the correlated IAPM, CWPM and MPM
    % methods. 
    %
    % Input:
    %   -   leni        : number of random numbers for each subject
    %   -   L           : number of random effects draws, 1 if isIAPM = F
    %   -   M           : number of subjects
    %   -   isIAPM      : indicator variable, true only if method is IAPM
    %   -   varargin    : m, current block being updated (can be a vector)
    %                     U, current vector of random numbers
    
    len = sum(leni);
    Ure = [];

    % generate initial random numbers
    if isempty(varargin)

        U = randn(len, 1);

        % if method is IAPM
        if isIAPM
            Ure = nrqmc(L*M, 2);
            Ure = Ure(:);
        end

    else
        %% block pseudo-marginal
        m = varargin{1};    % current block
        U = varargin{2};    % current vector of random numbers

        if isIAPM
            Ure = U(1:2*L*M);

            new_ure = nrqmc(L*length(m), 2);

            % X0
            S = 1 + L*(m-1);
            E = S + L - 1;
            Ure(S(1):E(end)) = new_ure(:, 1); 

            % Beta
            S = S + L*M;
            E = S + L - 1;
            Ure(S(1):E(end)) = new_ure(:, 2); 

            U = U((2*L*M+1):end);
        end
        
        % update random numbers for subject/s m
        S = 1 + sum(leni(1:(m(1)-1)))*(m(1)>1);
        E = S + sum(leni(m)) - 1;
        U(S:E) = randn(sum(leni(m)), 1);

    end

    newU = [Ure ; U];

end