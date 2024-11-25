function [out, bicCurve] = func_relax_bic(input, oneDimEstimator, maxDim, doBIC)
% use RELAX idea to estimate multi-dimensional parameters
% oneDimEstimator -- function estimating each dimension (output of the 
    % function should be a 2 element cell: results, recovered signal without noise)
% maxDim -- max possible dimension
% doBIC -- if BIC should be done
    % 0 -- do not use BIC
    % positive integer -- BIC factor
% out -- estimated result, cell
% bicCurve: variation of BIC value

if nargin == 3
    doBIC = 0;
end
input   = input(:);
N       = length(input); % input length
x0      = oneDimEstimator(input);
out     = cell(1, maxDim);
recvyInput = zeros(N, maxDim);
K       = length(x0); % number of parameters
out{1}  = reshape(x0{1}, 1, []);
recvyInput(:, 1)  = x0{2};

if doBIC
    bicCurve = zeros(1, maxDim);
    resInput = input - recvyInput(:, 1);
    varNoise = norm(resInput)^2/N;
    bicCurve(1) = N*log(varNoise) + (doBIC + 1)*log(N);
end

for dd = 2:maxDim
    out{dd} = zeros(dd, K);
    out{dd}(1: dd - 1, :) = out{dd - 1};
    y = input - sum(recvyInput(:, 1: dd-1), 2);
    x0 = oneDimEstimator(y);
    out{dd}(dd, :) = reshape(x0{1}, 1, []); 
    recvyInput(:, dd) = x0{2};

    isloop = 1; cnt = 0;
    while isloop
        cnt = cnt + 1;
        for kk = 1:dd
            y = input - sum(recvyInput, 2) + recvyInput(:, kk);
            x0 = oneDimEstimator(y);
            out{dd}(kk, :) = reshape(x0{1}, 1, []); 
            recvyInput(:, kk) = x0{2};
        end
        if cnt > 5
            isloop = 0;
        end
    end

    if doBIC
        resInput = input - sum(recvyInput(:, 1: dd), 2);
        varNoise = norm(resInput)^2/N;
        bicCurve(dd) = N*log(varNoise) + (dd * doBIC + 1)*log(N);
    end
end
if doBIC 
    [~, dimEst] = min(bicCurve);
else
    dimEst = maxDim;
end
out = out{dimEst};
end