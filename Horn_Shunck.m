N = 55; % размер квадратного изображения
frame_width = 1; % ширина рамки
shift = 1; % сдвиг
alpha = 1; % подбирается перебором
A1 = generation_imeg(N); % генерация изображения
A2 = right_shift(A1, N, shift); % создание смещенного на 1 пиксель вправо изображения
%--------создание рамки------------------------------
b1 = zeros(N, shift); 
A1 = [b1 A1 b1];
A2 = [b1 A2 b1];
b1 = zeros(shift, N+2*shift);
A1 = [b1; A1; b1];
A2 = [b1; A2; b1];
%----------------------------------------------------
mX = zeros(N, N); % производная rho_x
mY = zeros(N, N); % производная rho_y
mT = zeros(N, N); % производная rho_t
%----------------------------------------------------
for j = 1+shift:N+shift
    for i = 1+shift:N+shift
%----------------------------------------------------
% далее реализованы различные аппроксимации производных
% функции яркости
% перед запуском необходимо закоментировать лишние
%----------------------------------------------------
% центральные разности
%----------------------------------------------------
        mT(i-shift,j-shift) = double(A2(i,j) - A1(i,j)); % правые разности, т.к. рассматриваем только 2 изображения
        if j < N+shift && j > shift
            mX(i-shift,j-shift) = double(A1(i,j+1) - A1(i,j-1))/2;
        else
            if j == N+shift
                mX(i-shift,j-shift) = double(A1(i,j+1) - A1(i,j));
            else
                mX(i-shift,j-shift) = double(A1(i,j) - A1(i,j-1));
            end
        end
        if i < N+shift && i > shift
            mY(i-shift,j-shift) = double(A1(i+1,j) - A1(i-1,j))/2;
        else
            if i == N+shift
                mY(i-shift,j-shift) = double(A1(i+1,j) - A1(i,j));
            else
                mY(i-shift,j-shift) = double(A1(i,j) - A1(i-1,j));
            end
        end
%----------------------------------------------------
% левые разности
%----------------------------------------------------
        mT(i-shift,j-shift) = double(A2(i,j) - A1(i,j));
        if j > 1+shift
            mX(i-shift,j-shift) = double(A1(i,j) - A1(i,j-1));
        else
            mX(i-shift,j-shift) = double(A1(i,j+1) - A1(i,j));
        end
        if i < 1+shift
            mY(i-shift,j-shift) = double(A1(i,j) - A1(i-1,j));
        else
            mY(i-shift,j-shift) = double(A1(i+1,j) - A1(i,j));
        end
%----------------------------------------------------
% разности, с применением всех трех переменных (x, y, t)
%----------------------------------------------------
        if i < N+shift && j < N+shift % из статьи
            mX(i-shift,j-shift) = (1/4)*(double(A1(i,j+1)-A1(i,j)+A1(i+1,j+1)-A1(i+1,j)+A2(i,j+1)-A2(i,j)+A2(i+1,j+1)-A2(i+1,j)));
            mY(i-shift,j-shift) = (1/4)*(double(A1(i+1,j)-A1(i,j)+A1(i+1,j+1)-A1(i,j+1)+A2(i+1,j)-A2(i,j)+A2(i+1,j+1)-A2(i,j+1)));
            mT(i-shift,j-shift) = (1/4)*(double(A2(i,j)-A1(i,j)+A2(i+1,j)-A1(i+1,j)+A2(i,j+1)-A1(i,j+1)+A2(i+1,j+1)-A1(i+1,j+1)));
        else
            if i == N+shift && j == N+shift
                mX(i-shift,j-shift) = (1/4)*(double(A1(i-1,j)-A1(i-1,j-1)+A1(i,j)-A1(i,j-1)+A2(i-1,j)-A2(i-1,j-1)+A2(i,j)-A2(i,j-1)));
                mY(i-shift,j-shift) = (1/4)*(double(A1(i,j-1)-A1(i-1,j-1)+A1(i,j)-A1(i-1,j)+A2(i,j-1)-A2(i-1,j-1)+A2(i,j)-A2(i-1,j)));
                mT(i-shift,j-shift) = (1/4)*(double(A2(i-1,j-1)-A1(i-1,j-1)+A2(i,j-1)-A1(i,j-1)+A2(i-1,j)-A1(i-1,j)+A2(i,j)-A1(i,j)));
            else
                if i == N+shift
                    mX(i-shift,j-shift) = (1/4)*(double(A1(i-1,j+1)-A1(i-1,j)+A1(i,j+1)-A1(i,j)+A2(i-1,j+1)-A2(i-1,j)+A2(i,j+1)-A2(i,j)));
                    mY(i-shift,j-shift) = (1/4)*(double(A1(i,j)-A1(i-1,j)+A1(i,j+1)-A1(i-1,j+1)+A2(i,j)-A2(i-1,j)+A2(i,j+1)-A2(i-1,j+1)));
                    mT(i-shift,j-shift) = (1/4)*(double(A2(i-1,j)-A1(i-1,j)+A2(i,j)-A1(i,j)+A2(i-1,j+1)-A1(i-1,j+1)+A2(i,j+1)-A1(i,j+1)));
                end
                if j == N+shift
                    mX(i-shift,j-shift) = (1/4)*(double(A1(i,j)-A1(i,j-1)+A1(i+1,j)-A1(i+1,j-1)+A2(i,j)-A2(i,j-1)+A2(i+1,j)-A2(i+1,j-1)));
                    mY(i-shift,j-shift) = (1/4)*(double(A1(i+1,j-1)-A1(i,j-1)+A1(i+1,j)-A1(i,j)+A2(i+1,j-1)-A2(i,j-1)+A2(i+1,j)-A2(i,j)));
                    mT(i-shift,j-shift) = (1/4)*(double(A2(i,j-1)-A1(i,j-1)+A2(i+1,j-1)-A1(i+1,j-1)+A2(i,j)-A1(i,j)+A2(i+1,j)-A1(i+1,j)));
                end
            end
        end
%----------------------------------------------------
     end
end
xy = zeros(N^2,1); % диагональ матрицы B
k = 1;
for i = 1:N
    for j = 1:N
        xy(k) = mX(i,j)*mY(i,j);
        k = k + 1;
    end
end
A = zeros(N^2);
B = diag(xy);
k = 1;
%---------------------------------------------------
% далее реализованы заполнения матриц A и C в зависимости
% от аппроксимации лапласиана
% перед запуском необходимо закоментировать лишнее
%---------------------------------------------------
% по 5 точкам
%---------------------------------------------------
for i = 1:N
    for j = 1:N
        A((i-1)*N+j,(i-1)*N+j) = 4*alpha^2 + mX(i,j)^2;
        if i-1 > 0
            A(k, (i-2)*N+j) = -alpha^2;
        end
        if j-1 > 0
            A(k, (i-1)*N+j-1) = -alpha^2;
        end
        if i+1 <= N
            A(k, (i)*N+j) = -alpha^2;
        end
        if j+1 <= N
            A(k, (i-1)*N+j+1) = -alpha^2;
        end
        k = k+1;
    end
end
C = A;
for i = 1:N
    for j = 1:N
        C((i-1)*N+j,(i-1)*N+j) = 4*alpha^2 + mY(i,j)^2;
    end
end
%--------------------------------------------
% по 9 точкам
%--------------------------------------------
for i = 1:N
    for j = 1:N
        A((i-1)*N+j,(i-1)*N+j) = alpha^2 + mX(i,j)^2;
        if i-1 > 0 && j+1 <= N
            A(k, (i-2)*N+j+1) = -(1/12)*alpha^2;
        end
        if i+1 <= N && j+1 <= N
            A(k, (i)*N+j+1) = -(1/12)*alpha^2;
        end
        if i+1 <= N && j-1 > 0
            A(k, (i)*N+j-1) = -(1/12)*alpha^2;
        end
        if i-1 > 0 && j-1 > 0
            A(k, (i-2)*N+j-1) = -(1/12)*alpha^2;
        end
        if j+1 <= N
            A(k, (i-1)*N+j+1) = -(1/6)*alpha^2;
        end
        if i+1 <= N
            A(k, (i)*N+j) = -(1/6)*alpha^2;
        end
        if i-1 > 0
            A(k, (i-2)*N+j) = -(1/6)*alpha^2;
        end
        if j-1 > 0
            A(k, (i-1)*N+j-1) = -(1/6)*alpha^2;
        end
        k = k+1;
    end
end
C = A;
for i = 1:N
    for j = 1:N
        C((i-1)*N+j,(i-1)*N+j) = alpha^2 + mY(i,j)^2;
    end
end
%-----------------------------------------
q = zeros(2*N^2, 1); % вектор (d e)^T
k = 1;
for i = 1:N
    for j = 1:N
        q(k) = -mX(i,j)*mT(i,j);
        k = k + 1;
    end
end
for i = 1:N
    for j = 1:N
        q(k) = -mY(i,j)*mT(i,j);
        k = k + 1;
    end
end
vu = Seidel_method([A B; B C], q, 2*(N)^2);
u = zeros(N+2*shift);
v = u;
k = 1;
for i = 1+shift:N+shift
    for j = 1+shift:N+shift
        u(i,j) = vu(k);
        k = k + 1;
    end
end
for i = 1+shift:N+shift
    for j = 1+shift:N+shift
        v(i,j) = vu(k);
        k = k + 1;
    end
end
%-------построение поля скоростей-------------------
A3 = ones(N+2*shift);
figure(1)
imshow(A3);
hold on
for i = 2:N
    for j = 2:N
        Z = quiver(u(i,j)+i, v(i,j)+j, -u(i,j), -v(i,j), 'r');
        Z.ShowArrowHead = 'off';
        Z.Marker = '.';
        hold on
    end
end