function out = mdim_node_merge(y, func_steer, ini, err_rate)
% func_steer should be function of w
a   = ini.amp; 
K   = length(a);
a   = reshape(a, K, 1); % K x 1

w   = ini.freq; 
if size(w, 1) ~= K
    w = w.'; %size(w) = [K, dim]
end
y   = reshape(y, [], 1);

ismerge = 1;
while ismerge
    if length(a) == 1
        ismerge = 0;
    else
        [a, w, ismerge] = merge2node(y, func_steer, a, w, err_rate);
    end
end
out.amp = a;
out.freq = w;
end

function [a, w, ismerge] = merge2node(y, func_steer, a, w, err_rate)
[K, ndim] = size(w);
L  = length(y);
ismerge = 0;

baseEst = zeros(L, K);
for kk = 1:K
    baseEst(:, kk) = func_steer(w(kk, :));
end
noise = y - baseEst*a;
AhA = baseEst'*baseEst;
CovAmp = inv(AhA);

criterion = finv(1 - err_rate, 2, 2*(L - K))/(L-K);
criterion = norm(noise)^2*criterion;

wX = zeros(K^2-K, 2*ndim);
indset = zeros(K^2-K, 2);

for ii = 1:K
    seq = (ii-1)*(K-1)+1: ii*(K-1);
    wX(seq, 1:ndim) = repmat(w(ii,:), K-1,1);
    wX(seq, ndim+1:2*ndim) = w([1:ii-1,ii+1:end], :);
    indset(seq, 1)  = ii*ones(K-1,1);
    indset(seq, 2)  = [1:ii-1,ii+1:K].';
end

dw = zeros(K^2-K, 1);
for nn = 1:ndim
    dw = dw + angdiff(wX(:, nn), wX(:, nn+ndim)).^2;
end
[~, ind] = min(dw);
seq = indset(ind, :);

tmp1    = baseEst(:, seq)*a(seq);

wAvg    = abs(a(seq)).'*w(seq,:)/sum(abs(a(seq)));
baseAvg = func_steer(wAvg);
bb      = baseAvg'*baseAvg;
aAvg = baseAvg'*tmp1/bb;
tmp2 = baseAvg*aAvg;

Covij   = baseEst(:,seq)*CovAmp(seq, seq)*baseEst(:, seq)';
lambda  = real(trace(Covij) - baseAvg'*Covij*baseAvg/bb);
tmp     = norm(tmp1 - tmp2)^2/lambda;

if tmp <= criterion
    ismerge = 1;
    ind = setdiff(1:K, seq);
    w = [w(ind, :); wAvg];
    a = [a(ind, :); aAvg];
end
end