function p = logprior_sigma(sigma)
% Evaluates the prior for sigma, where sigma is not transformed

p = 0.5*log(2/pi)-log(5) - sigma^2/(2*5^2);

end