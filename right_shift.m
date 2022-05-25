function B = right_shift(A, n, a)
B = A(:,1:(n-a));
b = zeros(n, a);
B = [b B];