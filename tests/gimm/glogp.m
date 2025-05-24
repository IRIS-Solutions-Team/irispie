
function [mu, sigma, nu, low, high] = glogp(varargin)

    if numel(varargin)>=5
        high = varargin{5};
    else
        high = 1;
    end

    if numel(varargin)>=4
        low = varargin{4};
    else
        low = 0;
    end

    inxSwap = high<low;
    if any(inxSwap(:))
        [low(inxSwap), high(inxSwap)] = deal(high(inxSwap), low(inxSwap));
    end

    if numel(varargin)>=3
        nu = varargin{3};
    else
        nu = 0;
    end

    if numel(varargin)>=2
        sigma = abs(varargin{2});
    else
        sigma = 1;
    end

    if isempty(varargin)
        mu = 0;
    end

    % y0[x=0], sigma, nu, low-y0, high-y0
    if isempty(varargin)
        y0 = 0.5;
    else
        y0 = varargin{1};
    end
    low = y0 + low;
    high = y0 + high;
    mu = sigma .* log( ((y0-low)./(high-low)) .^ (-1./exp(nu)) - 1 );

end%

