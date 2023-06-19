function [x, loss] = func_BFGS(obj, grad, x0, options)
% obj -- cost function
% grad -- gradient function
% x0 --  initial value

if nargin == 3 || isempty(options)
    options.maxiter = 1e3;
    options.tol     = 1e-12;
    options.lb      = -inf(size(x0));
    options.ub      = inf(size(x0));
    options.wrap    = zeros(size(x0));
end

if isfield(options, 'maxiter') == 0
    options.maxiter = 1e3;
end

if isfield(options, 'tol') == 0
    options.tol     = 1e-12;
end

if isfield(options, 'lb') == 0
    options.lb = -inf(size(x0));
end

if isfield(options, 'ub') == 0
    options.ub = inf(size(x0));
end

if isfield(options, 'wrap') == 0
    options.wrap    = zeros(size(x0));              
end

wrap    = options.wrap;
alpha   = 1;
loss    = obj(x0);
K       = length(x0); % number of variables

% first iteration
g       = grad(x0); % calculate gradient
B       = eye(K);
d       = -B*g; % descent direction
[x, s]  = backtrack_line_search(obj, x0, alpha, g, d);

g0      = g;
B0      = B;
loss    = [loss, obj(x)];

isloop  = 1;
ii      = 1;
while isloop
    g   = grad(x); % calculate gradient
    y   = g - g0;
    B   = func_B(y, s, B0); % calculate quasi-inverse Hessian
    
    if sum(isnan(B)) > 0
        break;
    end
    d   = -B*g;
    
    [x, s] = backtrack_line_search(obj, x, alpha, g, d);
    x(find(x<= options.lb)) = options.lb(find(x<= options.lb));
    x(find(x>= options.ub)) = options.ub(find(x>= options.ub));
    x = wrapToPi(wrap.*x) + (~wrap).*x;          
    g0  = g;
    B0  = B;
    loss = [loss, obj(x)];

    ii  = ii + 1;
    if (ii >= options.maxiter) || ((g.'*g) < options.tol)
        isloop = 0;
    end
end
end

function [x, s] = backtrack_line_search(obj, x, alpha0, g, d)
rho     = 0.95;
c       = 0.1;
alpha   = alpha0;
value0  = obj(x);
value1  = c*g.'*d;
while (obj(x + alpha*d) > value0 + alpha*value1) 
    alpha = rho*alpha;
end
s = alpha*d;
x = x + s;
end

function B = func_B(y, s, B0)
tmp     = s.'*y;
B       = B0 + (tmp + y.'*B0*y)*(s*s.')./(tmp^2) - ...
    (B0*y*s.' + s*y.'*B0)/tmp;
end