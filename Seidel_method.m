function x2 = Seidel_method(A, b, n)
    epsilon = norm(A,Inf)*(10^(-6));
    B = A;
    C = b;
    for t = 1:n
        for j = 1:n
            if t ~= j
                B(t,j) = - B(t,j)/A(t,t);
            else
                B(t,j) = 0;
            end
        end
        C(t) = C(t)/A(t,t);
    end
    q = norm(B, Inf);
    x_0 = C;
    x_1 = x_0;
    for t = 1:n
        x_1(t) = C(t);
        for j = 1:n
            x_1(t) = x_1(t) + x_1(j)*B(t,j);
        end
    end
    x = x_1;
    for t = 1:n
        x(t) = C(t);
        for j = 1:n
            x(t) = x(t) + x(j)*B(t,j);
        end
    end
    k = 2; % так как подсчитали x_1 и x
    while norm(x_1-x_0,Inf) > abs(epsilon*(1-q)/q) 
        x_0 = x_1;
        x_1 = x;
        x = x_1;
        for t = 1:n
            x(t) = C(t);
            for j = 1:n
                x(t) = x(t) + x(j)*B(t,j);
            end
        end
        k = k + 1;
    end
    x2 = x;
end
