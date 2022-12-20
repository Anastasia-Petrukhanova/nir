from PIL import Image, ImageDraw
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy import array,inf
import copy
import math

def revers_m(M):
  det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
  return np.array([[M[1,1]/det, -M[0,1]/det], [-M[1,0]/det, M[0,0]/det]])

def S_block(N, b, H):
    x_0 = np.array(b.copy())
    x_1 = np.array(b.copy())
    eps = 10**(-6)
    print(eps)
    for i in range(0, N**2):
        if i % N == 0:
            x_1[i*2:i*2+2] = revers_m(H[i*2:i*2+2, i*2:i*2+2]).dot(-H[i*2:i*2+2, (i+1)*2:(i+1)*2+2].dot(x_1[(i+1)*2:(i+1)*2+2]) + b[i*2:i*2+2])
        elif (i+1) % N == 0:
            x_1[i*2:i*2+2] = revers_m(H[i*2:i*2+2, i*2:i*2+2]).dot(-H[i*2:i*2+2, (i-1)*2:(i-1)*2+2].dot(x_1[(i-1)*2:(i-1)*2+2]) + b[i*2:i*2+2])
        else:
            x_1[i*2:i*2+2] = revers_m(H[i*2:i*2+2, i*2:i*2+2]).dot(-H[i*2:i*2+2, (i-1)*2:(i-1)*2+2].dot(x_1[(i-1)*2:(i-1)*2+2]) - H[i*2:i*2+2, (i+1)*2:(i+1)*2+2].dot(x_1[(i+1)*2:(i+1)*2+2]) + b[i*2:i*2+2])

        if i >= 0 and i <= N**2 - N - 1:
            x_1[i*2:i*2+2] += revers_m(H[i*2:i*2+2, i*2:i*2+2]).dot(-H[i*2:i*2+2, (N+i)*2:(N+i)*2+2].dot(x_1[(N+i)*2:(N+i)*2+2]))
        if i >= N:
            x_1[i*2:i*2+2] += revers_m(H[i*2:i*2+2, i*2:i*2+2]).dot(-H[i*2:i*2+2, (i-N)*2:(i-N)*2+2].dot(x_1[(i-N)*2:(i-N)*2+2]))

    n1 = x_1.copy()
    n1 = n1 - x_0
    x_0 = x_1.copy()
    k = 1
    while norm(n1,inf) > eps:
        print("norm = ", norm(n1,inf), "\t k = ", k)
        for i in range(0, N**2):
            if i % N == 0:
                x_1[i*2:i*2+2] = revers_m(H[i*2:i*2+2, i*2:i*2+2]).dot(-H[i*2:i*2+2, (i+1)*2:(i+1)*2+2].dot(x_1[(i+1)*2:(i+1)*2+2]) + b[i*2:i*2+2])
            elif (i+1) % N == 0:
                x_1[i*2:i*2+2] = revers_m(H[i*2:i*2+2, i*2:i*2+2]).dot(-H[i*2:i*2+2, (i-1)*2:(i-1)*2+2].dot(x_1[(i-1)*2:(i-1)*2+2]) + b[i*2:i*2+2])
            else:
                x_1[i*2:i*2+2] = revers_m(H[i*2:i*2+2, i*2:i*2+2]).dot(-H[i*2:i*2+2, (i-1)*2:(i-1)*2+2].dot(x_1[(i-1)*2:(i-1)*2+2]) - H[i*2:i*2+2, (i+1)*2:(i+1)*2+2].dot(x_1[(i+1)*2:(i+1)*2+2]) + b[i*2:i*2+2])

            if i >= 0 and i <= N**2 - N - 1:
                x_1[i*2:i*2+2] += revers_m(H[i*2:i*2+2, i*2:i*2+2]).dot(-H[i*2:i*2+2, (N+i)*2:(N+i)*2+2].dot(x_1[(N+i)*2:(N+i)*2+2]))
            if i >= N:
                x_1[i*2:i*2+2] += revers_m(H[i*2:i*2+2, i*2:i*2+2]).dot(-H[i*2:i*2+2, (i-N)*2:(i-N)*2+2].dot(x_1[(i-N)*2:(i-N)*2+2]))

        n1 = x_1.copy()
        for t in range(0, len(x_1)):
            n1[t] = n1[t] - x_0[t]

        k += 1
        x_0 = x_1.copy()

    return x_1

def random_img(size, frame_width):
    A = []
    for j in range(0, size[1] + 2*frame_width):
        a = []
        for i in range(0, size[0]+2*frame_width):
            a.append(randint(0, 10000)/10000)
        A.append(a)
    return A

def right_diag_shift(A, size, shift):
    B = []
    for j in range(0, size[1]):
        b =[]
        for i in range(0, size[0]):
            if i < shift or j >= (size[1] - shift):
                b.append(0)
            else:
                b.append(A[j+shift][i-shift])
        B.append(b)
    return B

def xyz_differences(N):
    for j in range(frame_width, N+frame_width):
        for i in range(frame_width, N+frame_width):
            if i < N+frame_width-1 and j < N+frame_width-1:
                mX[i-frame_width][j-frame_width] = (1/4)*(A1[i][j+1]-A1[i][j]+A1[i+1][j+1]-A1[i+1][j]+A2[i][j+1]-A2[i][j]+A2[i+1][j+1]-A2[i+1][j])
                mY[i-frame_width][j-frame_width] = (1/4)*(A1[i+1][j]-A1[i][j]+A1[i+1][j+1]-A1[i][j+1]+A2[i+1][j]-A2[i][j]+A2[i+1][j+1]-A2[i][j+1])
                mT[i-frame_width][j-frame_width] = (1/4)*(A2[i][j]-A1[i][j]+A2[i+1][j]-A1[i+1][j]+A2[i][j+1]-A1[i][j+1]+A2[i+1][j+1]-A1[i+1][j+1])
            else:
                if i == N+frame_width-1 and j == N+frame_width-1:
                    mX[i-frame_width][j-frame_width] = (1/4)*(A1[i-1][j]-A1[i-1][j-1]+A1[i][j]-A1[i][j-1]+A2[i-1][j]-A2[i-1][j-1]+A2[i][j]-A2[i][j-1])
                    mY[i-frame_width][j-frame_width] = (1/4)*(A1[i][j-1]-A1[i-1][j-1]+A1[i][j]-A1[i-1][j]+A2[i][j-1]-A2[i-1][j-1]+A2[i][j]-A2[i-1][j])
                    mT[i-frame_width][j-frame_width] = (1/4)*(A2[i-1][j-1]-A1[i-1][j-1]+A2[i][j-1]-A1[i][j-1]+A2[i-1][j]-A1[i-1][j]+A2[i][j]-A1[i][j])
                else:
                    if i == N+frame_width-1:
                        mX[i-frame_width][j-frame_width] = (1/4)*(A1[i-1][j+1]-A1[i-1][j]+A1[i][j+1]-A1[i][j]+A2[i-1][j+1]-A2[i-1][j]+A2[i][j+1]-A2[i][j])
                        mY[i-frame_width][j-frame_width] = (1/4)*(A1[i][j]-A1[i-1][j]+A1[i][j+1]-A1[i-1][j+1]+A2[i][j]-A2[i-1][j]+A2[i][j+1]-A2[i-1][j+1])
                        mT[i-frame_width][j-frame_width] = (1/4)*(A2[i-1][j]-A1[i-1][j]+A2[i][j]-A1[i][j]+A2[i-1][j+1]-A1[i-1][j+1]+A2[i][j+1]-A1[i][j+1])
                    if j == N+frame_width-1:
                        mX[i-frame_width][j-frame_width] = (1/4)*(A1[i][j]-A1[i][j-1]+A1[i+1][j]-A1[i+1][j-1]+A2[i][j]-A2[i][j-1]+A2[i+1][j]-A2[i+1][j-1])
                        mY[i-frame_width][j-frame_width] = (1/4)*(A1[i+1][j-1]-A1[i][j-1]+A1[i+1][j]-A1[i][j]+A2[i+1][j-1]-A2[i][j-1]+A2[i+1][j]-A2[i][j])
                        mT[i-frame_width][j-frame_width] = (1/4)*(A2[i][j-1]-A1[i][j-1]+A2[i+1][j-1]-A1[i+1][j-1]+A2[i][j]-A1[i][j]+A2[i+1][j]-A1[i+1][j])

size = [100,100] # изменяемые данные (только квадратное изображение, без учета границ)
step = 10 # шаг, с которым будут отображены векторы (можно поменять и тогда отобразится больше векторов)
N = size[0]
frame_width = 1
shift = 1
alpha = 1 # параметр регуляризации по А. Н. Тихонову , который также можно изменить

A1 = []
A2 = []

A1 = random_img(size, frame_width)
A2 = right_diag_shift(A1, [size[0]+2*frame_width, size[1]+2*frame_width], shift)
plt.imshow(A1, cmap="Greys")
plt.show()
plt.imshow(A2, cmap="Greys")
plt.show()

A3 = a = [[255] * (size[0]+2*frame_width)] * (size[1]+2*frame_width)
A3 = Image.fromarray(np.array(A3))

accur0 = []
accur1 = []
accur2 = []

for count in [0,1,2]:
    mX = []
    q = []
    u = []
    v = []
    for j in range(0, size[1] + 2*frame_width):
        U = []
        V = []
        for i in range(0, size[0] + 2*frame_width):
            if count == 0:
                U.append(0)
                V.append(0)
            elif count == 1:
                U.append(1)
                V.append(-1)
            else:
                U.append(1)
                V.append(0)
        u.append(U)
        v.append(V)
    for j in range(0, size[1]):
        x = []
        for i in range(0, size[0]):
            x.append(0)
        mX.append(x)
    mY = copy.deepcopy(mX)
    mT = copy.deepcopy(mX)
    xyz_differences(size[0])
    a = []
    for j in range(0, size[1]):
        for i in range(0, size[0]):
            a.append(mX[j][i]*mY[j][i])

    for i in range(0, size[1]):
        for j in range(0, size[0]):
            q.append([-mX[i][j]*mT[i][j]])
            q.append([-mY[i][j]*mT[i][j]])
            if i == 0:
                q[-2][0] += (alpha**2)*u[i][j+1]
                q[-1][0] += (alpha**2)*v[i][j+1]
            if i == size[0]-1:
                q[-2][0] += (alpha**2)*u[-1][j+1]
                q[-1][0] += (alpha**2)*v[-i][j+1]
            if j == 0:
                q[-2][0] += (alpha**2)*u[i+1][j]
                q[-1][0] += (alpha**2)*v[i+1][j]
            if j == size[0]-1:
                q[-2][0] += (alpha**2)*u[i+1][-1]
                q[-1][0] += (alpha**2)*v[i+1][-1]
    q = np.array(q)
    mX = np.array(mX)
    mY = np.array(mY)
    mT = np.array(mT)
    M = []
    for i in range(0, 2*size[0]*size[1]):
        M.append([0.0]*(2*size[0]*size[1]))
    M = np.array(M)
    for i in range(0, 2*size[0]*size[1]):
        if i % 2 == 0:
            M[i, i] = 4*alpha*alpha + mX[(i//2)//N, (i//2)%N]*mX[(i//2)//N, (i//2)%N]
            M[i, i+1] = a[i//2]
        else:
            M[i, i] = 4*alpha*alpha + mY[((i-1)//2)//N, (i//2)%N]*mY[((i-1)//2)//N, ((i-1)//2)%N]
            M[i, i-1] = a[(i-1)//2]
        if (i) % (size[0]*2) == 0 or (i-1) % (size[0]*2) == 0:
            M[i, i+2] = -alpha*alpha
        elif (i+1) % (size[0]*2) == 0 or (i+2) % (size[0]*2) == 0:
            M[i, i-2] = -alpha*alpha
        else:
            M[i, i+2] = -alpha*alpha
            M[i, i-2] = -alpha*alpha
        if i < size[0]*2 or i < 2*size[0]**2-size[0]*2:
            M[i, i+2*size[0]] = -alpha**2
        if i >= size[0]*2 or i > 2*size[0]**2-size[0]*2:
            M[i, i - 2*size[0]] = -alpha**2

    vu = S_block(size[0], q, M)

    tan = []
    for j in range(0, size[1] + 2*frame_width):
        t = []
        for i in range(0, size[0] + 2*frame_width):
            t.append(0)
        tan.append(t)
    k = 0
    for j in range(frame_width, size[0]+frame_width):
        for i in range(frame_width, size[1]+frame_width):
            u[i][j] = vu[k, 0]
            v[i][j] = vu[k+1, 0]
            k = k + 2

    for j in range(frame_width, size[0]+frame_width):
        for i in range(frame_width, size[1]+frame_width):
            tan[i][j] = abs(-1-v[i][j]/u[i][j])

    ax = plt.figure().gca()
    ax.imshow(A3, cmap = 'gray')

    for i in range(0, size[0] + 2*frame_width, step):
        for j in range(0, size[1] + 2*frame_width, step):
            ax.arrow(i, j, step*u[j][i], step*v[j][i], length_includes_head = True, head_width = 0.15*step, overhang = 1, head_length = 0.2*step, width = 0.01)
    plt.draw()
    plt.show()

    ax = plt.figure().gca()
    ax.imshow(A3, cmap = 'gray')

    for i in range(0, size[0] + 2*frame_width, step):
        for j in range(0, size[1] + 2*frame_width, step):
            c = math.sqrt(u[j][i]**2+v[j][i]**2)
            if c != 0:
                ax.arrow(i, j, step*u[j][i]/c, step*v[j][i]/c, length_includes_head = True, head_width = 0.15*step, overhang = 1, head_length = 0.2*step, width = 0.01)
            else:
                ax.arrow(i, j, step*u[j][i], step*v[j][i], length_includes_head = True, head_width = 0.15*step, overhang = 1, head_length = 0.2*step, width = 0.01)
    plt.draw()
    plt.show()

    frame_w = []
    for t in range(0, size[0]):
        sum = 0
        for j in range(frame_width+t, size[0]+frame_width-t):
            for i in range(frame_width+t, size[1]+frame_width-t):
                sum += tan[i][j]
        if count == 0:
            accur0.append(sum/((size[0]-t)*(size[1]-t)))
        elif count == 1:
            accur1.append(sum/((size[0]-t)*(size[1]-t)))
        else:
            accur2.append(sum/((size[0]-t)*(size[1]-t)))
        frame_w.append(t)


fig, ax = plt.subplots()

plt.plot(frame_w, accur0, color='r', label='граничные условия 0')
plt.plot(frame_w, accur1, color='b', label='граничные условия 1')
plt.plot(frame_w, accur2, color='g', label='граничные условия 1, 0')
plt.xlabel('Расстояние до границы')
plt.legend()

plt.show()
