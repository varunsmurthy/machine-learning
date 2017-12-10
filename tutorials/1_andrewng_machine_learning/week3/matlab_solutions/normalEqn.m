
function [theta] = normalEqn(X, y)

theta = pinv(transpose(X)*X)*transpose(X)*y;