function [x, loss] = func_conjugate_gradient(func, gradient, x0, update_method, options)
% func  -- cost function
% gradient -- gradient function
% x0 -- initial value

if nargin == 4 || isempty(options)
    options.maxiter = 1e3;
    options.tol     = 1e-6;
    options.lb      = -inf(size(x0));
    options.ub      = inf(size(x0));
    options.wrap    = zeros(size(x0));
end

if isfield(options, 'maxiter') == 0
    options.maxiter = 1e3;
end

if isfield(options, 'tol') == 0
    options.tol     = 1e-6;
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


tol         = options.tol; % tolerance
wrap        = options.wrap;
maxiter     = options.maxiter;
alpha       = 1;
loss        = func(x0);

% choose update parameter
switch update_method
    case 'HS'
        func_beta = str2func('func_HS');
    case 'FR'
        func_beta = str2func('func_FR');
    case 'PRP'
        func_beta = str2func('func_PRP');
    case 'CD'
        func_beta = str2func('func_CD');
    case 'LS'
        func_beta = str2func('func_LS');
    case 'DY'
        func_beta = str2func('func_DY');
    case 'HZ'
        func_beta = str2func('func_HZ');
    case 'HZ2'
        func_beta = str2func('func_HZ2');
    case 'DYHS'
        func_beta = str2func('func_DYHS');
end

% first iteration
g   = gradient(x0); % calculate gradient
d   = -g; % descent direction
alpha0 = alpha; % initial step length
% backtrack line search and update
[x, alpha] = backtrack_line_search(func, x0, alpha0, g, d);

d0 = d; g0 = g;
loss = [loss, func(x)];

isloop = 1;
ii = 1;
while isloop
    g = gradient(x); % calculate gradient;
    beta = func_beta(g, g0, d0); % calculate update parameter
    d = -g + beta*d0; % update direction
    alpha0 = alpha*(g0.'*d0)/(g.'*d); % initialize step length
    % backtrack line search and update
    [x, alpha] = backtrack_line_search(func, x, alpha0, g, d);
    x(find(x<= options.lb)) = options.lb(find(x<= options.lb));
    x(find(x>= options.ub)) = options.ub(find(x>= options.ub));
    x = wrapToPi(wrap.*x) + (~wrap).*x; 

    d0 = d; g0 = g;
    loss = [loss, func(x)];

    ii = ii + 1;
    if (ii>=maxiter) || ((g.'*g) < tol)
        isloop = 0;
    end
end
end

function [x, alpha] = backtrack_line_search(func, x, alpha0, g, d)
rho     = 0.9;
c       = 0.1;
alpha   = alpha0;
value0  = func(x);
value1  = c*g.'*d;
while (func(x + alpha*d) > value0 + alpha*value1) 
    alpha = rho*alpha;
end
x = x + alpha*d;
end

function beta = func_HS(g, g0, d0)
s = g - g0;
beta = (g.'*s)/(d0.'*s);
end

function beta = func_FR(g, g0, d0)
beta = (g.'*g)/(g0.'*g0);
end

function beta = func_PRP(g, g0, d0)
s = g - g0;
beta = (g.'*s)/(g0.'*g0);
beta = max(beta, 0);
end

function beta = func_CD(g, g0, d0)
beta = (g.'*g)/(-d0.'*g0);
end

function beta = func_LS(g, g0, d0)
s = g - g0;
beta = (g.'*s)/(-d0.'*g0);
end

function beta = func_DY(g, g0, d0)
s = g - g0;
beta = (g.'*g)/(d0.'*s);
end

function beta = func_HZ(g, g0, d0)
s = g - g0;
beta = ((s - 2*d0*(s.'*s)/(d0.'*s)).'*g)/(d0.'*s);
end

function beta = func_HZ2(g, g0, d0)
s = g - g0;
beta = ((s - 2*d0*(s.'*s)/(d0.'*s)).'*g)/(d0.'*s);
eta = 0.01;
tmp = norm(d0)*min(eta, norm(g0));
eta = -1./tmp;
beta = max(beta, eta);
end

function beta = func_DYHS(g, g0, d0)
beta1 = func_DY(g, g0, d0);
beta2 = func_HS(g, g0, d0);
beta = min(beta1, beta2);
beta = max(beta, 0);
end