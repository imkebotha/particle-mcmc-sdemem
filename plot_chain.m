function plot_chain(chain) 
% Displays trace, density and autocorrelation plots for theta
% 
% Input: 
%   -   chain   : MCMC chain for theta
    
GAMMA = 1; SIGMA = 2; RHO = 3; MU_X0 = 4; SIGMA_X0 = 5; MU_BETA = 6; SIGMA_BETA = 7; 
N = size(chain, 2);
    
names = {'\gamma';'\sigma';'\rho';'\mu_{v0}';'\sigma_{v0}';'\mu_{\beta}';'\sigma_{\beta}'};
HN = @(y, s) sqrt(2/(s^2*pi))*exp(-y.^2/(2*s^2)); % half normal distribution

% density of priors
pr{GAMMA} = @(y) HN(y, 5); 
pr{SIGMA} = @(y) HN(y, 5);
pr{RHO} = @(y) normpdf(y, 1, 0.5); 
pr{MU_X0} = @(y) normpdf(y, 3, 4);
pr{SIGMA_X0} = @(y) HN(y, 5);
pr{MU_BETA} = @(y) normpdf(y, 0, 4);
pr{SIGMA_BETA} = @(y) HN(y, 5);
  

%%% TRACE plots
figure
for i = 1:N
    subplot(round(N/2), 2, i);
    plot(chain(:, i)); 

    title(names(i), 'FontSize', 14)

    xt = get(gca, 'XTick');
    set(gca, 'XTick', xt, 'XTickLabel', xt/1000);
end


%%% DENSITY plots
figure
for i = 1:N
    subplot(round(N/2), 2, i);      

    hold on
    [y, x] = ksdensity(chain(:, i));   
    plot(x, y, 'LineWidth', 1)

    if (i == 3)
        x_min = round(min(x)-0.01, 2) ;
        x_max = round(max(x)+0.01, 2) ;
    else
        x_min = round(min(x)-0.1, 1) - 0.1;
        x_max = round(max(x)+0.1, 1) + 0.1;
    end

    x = x_min:0.01:x_max;
    y = pr{i}(x);
    plot(x, y, 'LineWidth', 1)
    xlim([x_min x_max]);
    
    hold off

    title(names(i), 'FontSize', 14)
end


%%% AUTOCORRELATION plots
figure
for i = 1:N
    subplot(round(N/2), 2, i);      
    [acf, lags, bounds] = autocorr(chain(:, i));
    stem(lags, acf, 'filled', 'Marker', '.'); 
    hold on;
    h = line(lags, bounds(1)*ones(length(acf), 1));
    h1 = line(lags, bounds(2)*ones(length(acf), 1));
    set(h, 'color', [1 0 0]);
    set(h1, 'color', [1 0 0]);
    title(names(i), 'FontSize', 14)
end
    
end
