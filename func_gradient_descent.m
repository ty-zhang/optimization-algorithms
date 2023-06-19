function [x, loss] = func_gradient_descent(func, grad, x0, options)
% func -- cost function
% grad -- gradient function
% x0 -- initial value

if nargin == 3 || isempty(options)
    options.maxiter = 1e8;
    options.tol = 1e-6;
    options.stepsize = 1e-4;
    options.momentum_rate = 0;
    options.lb      = -inf(size(x0));
    options.ub      = inf(size(x0));
    options.wrap    = zeros(size(x0));
    options.fixstepsize = 1;
end

if isfield(options, 'maxiter') == 0
    options.maxiter = 1e8;
end

if isfield(options, 'tol') == 0
    options.tol     = 1e-6;
end

if isfield(options, 'stepsize') == 0
    options.stepsize     = 1e-4;
end

if isfield(options, 'momentum_rate') == 0
    options.momentum_rate     = 0;
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

if isfield(options, 'fixstepsize') == 0
    options.fixstepsize = 1;
end

mr      = options.momentum_rate; % momentum_rate
ss      = options.stepsize; % stepsize
tol     = options.tol; % tolerance
maxiter = options.maxiter; 
wrap    = options.wrap;

x       = x0;
dx      = 0;

isloop  = 1;
loss    = func(x);
ii      = 1;

alpha   = 1;
while isloop
    g   = grad(x); % calculate gradient
    dx  = mr*dx + (1 - mr)*g;

    if  (mr ~= 0) || (options.fixstepsize == 1)
        x   = x - ss*dx;
    else
        x = backtrack_line_search(func, x, alpha, g, -dx);
    end
    
    x(find(x<= options.lb)) = options.lb(find(x<= options.lb));
    x(find(x>= options.ub)) = options.ub(find(x>= options.ub));
    x = wrapToPi(wrap.*x) + (~wrap).*x; 
    
    loss = [loss, func(x)];
    if (norm(dx) < tol) || ii>= maxiter
        isloop = 0;
    end
    if mod(ii, 1e2) == 0 || ii == 1
        if (abs(loss(ii+1) - loss(ii)) < tol) || isnan(loss(ii+1))
            isloop = 0;
        end
    end

    ii = ii + 1;
end
end

function x = backtrack_line_search(obj, x, alpha, g, d)
rho     = 0.5;
c       = 0.1;
value0  = obj(x);
value1  = c*g.'*d;
while (obj(x + alpha*d) > value0 + alpha*value1) 
    alpha = rho*alpha;
end
s = alpha*d;
x = x + s;
end