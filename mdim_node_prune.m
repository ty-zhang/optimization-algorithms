function out = mdim_node_prune(y, func_steer, ini, err_rate)
% func_steer should be function of w
a   = ini.amp;
K   = length(a);
a   = reshape(a, K, 1); % K x 1

w   = ini.freq; 
if size(w, 1) ~= K
    w = w.'; %size(w) = [K, dim]
end
y   = reshape(y, [], 1);

isprune = 1;

while isprune
    [a, w, isprune] = prune1node(y, func_steer, a, w, err_rate);

    if isempty(a)
        isprune = 0;
    end
end
out.amp = a;
out.freq = w;
end

function [a, w, isprune] = prune1node(y, func_steer, a, w, err_rate)
K   = length(a);
L   = length(y); % signal length

isprune = 0;
criterion = finv((1-err_rate), 2, 2*(L-K))/(L - K);

baseEst = zeros(L, K);

for kk = 1:K
    baseEst(:, kk) = func_steer(w(kk, :));
end

noise = y - baseEst*a;
AhA = baseEst'*baseEst;
CovAmp = inv(AhA);

[~, ind] = min(abs(a));
lambda = CovAmp(ind, ind)*AhA(ind, ind);

tmp = norm(baseEst(:, ind)*a(ind))^2/norm(noise)^2;
tmp = tmp/lambda;

if tmp < criterion
    a = a([1:ind-1, ind+1:end]);
    w = w([1:ind-1, ind+1:end], :);
    isprune = 1;
end
end