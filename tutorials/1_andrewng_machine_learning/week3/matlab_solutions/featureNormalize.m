
function [X_norm, mu, sigma] = featureNormalize(X)

feature_count = size(X, 2);

mu = mean(X);
sigma = std(X);
X_norm = (X - mu)./(sigma);

end