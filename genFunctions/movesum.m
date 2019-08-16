function y = movesum(V, T)
    % returns moving sum of vector V with sum windows given by T. Similar
    % to movsum, except T is a vector such that length(V) = sum(T).
    y = zeros(1, length(T));

    ind = 1;
    for i = 1:length(T)
        y(i) = sum(V(ind:(ind+T(i)-1)));
        ind = ind + T(i);
    end
end