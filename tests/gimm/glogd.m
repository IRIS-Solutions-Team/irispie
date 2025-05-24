
function dy = glogd(x, varargin)

    [mu, sigma, nu, low, high] = glogp(varargin{:});

    nu1 = exp(nu);

    sigma1 = 1./sigma;

    z = (x - mu) .* sigma1;

    dy = nu1 .* sigma1 .* (1 + exp(-z)).^(-nu1-1) .* exp(-z);

    if any(low~=0) || any(high~=1)
        dy = (high - low) .* dy;
    end

end%

