
function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
parameter_count = size(theta,1);
temp_theta = theta;

for iter = 1:num_iters
    for j=1:parameter_count
        temp_theta(j) = theta(j) - (alpha.*(sum((X*theta - y).*X(:,j))))./(m);
    end

    %update theta with the newly calculated theta
    theta = temp_theta;
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end
end