function [ indx ] = resampleMultinomial(w , u , N)
% Performs multinomial resampling with replacement using prespecified
% uniform(0, 1) random numbers. Returns the index of the resampled points.
% 
% Input:
%   -   w   : normalised weights
%   -   u   : vector of U(0, 1) random numbers
%   -   N   : number of points to resample

u = normcdf(u, 0, 1);

Q = cumsum(w);
Q(end)=1; 

indx = zeros(N, 1);
i=1;
while (i<=N) 
    sampl = u(i); 
    j=1;
    while (Q(j)<sampl)
        j=j+1;
    end
    indx(i)=j;
    i=i+1;
end