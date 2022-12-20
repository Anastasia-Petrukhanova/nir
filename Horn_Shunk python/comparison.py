from PIL import Image, ImageDraw
import PIL
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy import array,inf
import copy
import math
import time

def revers_m(M):
  det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
  return np.array([[M[1,1]/det, -M[0,1]/det], [-M[1,0]/det, M[0,0]/det]])

def S_block(N, b, H):
    x_0 = np.array(b.copy())
    x_1 = np.array(b.copy())
    eps = 10**(-6)
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


step = 10 # шаг, с которым будут отображаться векторы (можно поменять и тогда отобразится больше векторов)
frame_width = 1
shift = 1
alpha = 1 # параметр регуляризации по А. Н. Тихонову , который также можно изменить

A1 = []
A2 = []

# Далее один из нужных вариантов - раскомментировать

#-----------генерация изображения с одинаковым сдвигом всех пикселей---------
size = [100,100]# изменяемая величина (только квадратное изображение, без учета границ)
A1 = random_img(size, frame_width)
A2 = right_diag_shift(A1, [size[0]+2, size[1]+2], shift)
#----------------------------------------------------------------------------

#------------------------готовое изображение---------------------------------
"""filename1 = "31.png"
with Image.open(filename1) as img1:
    img1.load()
filename2 = "32.png"
with Image.open(filename2) as img2:
    img2.load()

img1 = img1.convert("L")
img2 = img2.convert("L")
size = [0,0]
size[0], size[1] = img1.size

size[0] = size[0] - 2*frame_width
size[1] = size[1] - 2*frame_width

pix1 = img1.load()
pix2 = img2.load()
for j in range(0, size[1]+2*frame_width):
    a_1 = []
    a_2 = []
    #for i in range(0, size[0]+2*frame_width):
    for i in range(0, size[0]+2*frame_width):
        #if j < frame_width or j > size[1] or i < frame_width or i > size[0]:
        #    a.append(0)
        #else:
             a_1.append(pix1[i, j])
             a_2.append(pix2[i, j])
    A1.append(a_1)
    A2.append(a_2)"""
#----------------------------------------------------------------------------


ax = plt.figure().gca()
ax.imshow(A1, cmap = 'gray')
plt.draw()
plt.show()

ax = plt.figure().gca()
ax.imshow(A2, cmap = 'gray')
plt.draw()
plt.show()

A3 = [[255] * (size[0]+2*frame_width)] * (size[1]+2*frame_width)
A3 = Image.fromarray(np.array(A3))

N = size[0]
accur0 = []
accur1 = []

tic = time.perf_counter()
mX = []
q = []
u = []
v = []
for j in range(0, size[1] + 2*frame_width):
    U = []
    V = []
    for i in range(0, size[0] + 2*frame_width):
        U.append(0.0)
        V.append(0.0)
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
tic2 = time.perf_counter()
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
time_m = time.perf_counter()- tic

vu = S_block(size[0], q, M)



u = np.array(u)
v = np.array(v)
k = 0
for i in range(frame_width, size[0]+frame_width):
    for j in range(frame_width, size[1]+frame_width):
        u[j, i] = vu[k, 0]
        v[j, i] = vu[k+1, 0]
        k = k + 2
time1 = time.perf_counter() - tic

#--------------------для готового изображения закомметировать------------------
tan = []
for j in range(0, size[1] + 2*frame_width):
    t = []
    for i in range(0, size[0] + 2*frame_width):
        t.append(0)
    tan.append(t)

for j in range(frame_width, size[0]+frame_width):
    for i in range(frame_width, size[1]+frame_width):
        tan[i][j] = abs(-1-v[i][j]/u[i][j])
#------------------------------------------------------------------------


ax = plt.figure().gca()
ax.imshow(A3, cmap = 'gray')

for i in range(0, size[0] + 2*frame_width, step):
    for j in range(0, size[1] + 2*frame_width, step):
        ax.arrow(i, j, step*u[i, j], step*v[i, j], length_includes_head = True, head_width = 0.3*step, overhang = 1.5, head_length = 0.3*step, width = 0.01)

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

#--------------------для готового изображения закомметировать------------------
frame_w = []
for t in range(0, size[0]):
    sum = 0
    for j in range(frame_width+t, size[0]+frame_width-t):
        for i in range(frame_width+t, size[1]+frame_width-t):
            sum += tan[i][j]
    accur0.append(sum/((size[0]-t)*(size[1]-t)))
    frame_w.append(t)
#-------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------------------------------
im1 = Image.fromarray(np.array(A1))
im2 = Image.fromarray(np.array(A2))


from scipy.ndimage.filters import convolve as filter2
import os
from argparse import ArgumentParser

#compute magnitude in each 8 pixels. return magnitude average
def get_magnitude(u, v):
    scale = 3
    sum = 0.0
    counter = 0.0

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            counter += 1
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            sum += magnitude

    mag_avg = sum / counter

    return mag_avg



def draw_quiver(u,v,beforeImg):
    scale = 3
    ax = plt.figure().gca()
    ax.imshow(A3, cmap = 'gray')

    magnitudeAvg = get_magnitude(u, v)

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            #draw only significant changes
            if magnitude > magnitudeAvg:
                ax.arrow(j,i, dx, dy, length_includes_head = True, head_width = 0.3*step, overhang = 1.5, head_length = 0.3*step, width = 0.01)

    plt.draw()
    plt.show()



#compute derivatives of the image intensity values along the x, y, time
def get_derivatives(img1, img2):
    #derivative masks
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.25
    t_kernel = np.ones((2, 2)) * 0.25

    fx = filter2(img1,x_kernel) + filter2(img2,x_kernel)
    fy = filter2(img1, y_kernel) + filter2(img2, y_kernel)
    ft = filter2(img1, -t_kernel) + filter2(img2, t_kernel)

    return [fx,fy, ft]



#input: images name, smoothing parameter, tolerance
#output: images variations (flow vectors u, v)
#calculates u,v vectors and draw quiver
def computeHS(name1, name2, alpha, delta):
    #path = os.path.join(os.path.dirname(__file__), 'test images')
    #beforeImg = cv2.imread(os.path.join(path, name1), cv2.IMREAD_GRAYSCALE)
    #afterImg = cv2.imread(os.path.join(path, name2), cv2.IMREAD_GRAYSCALE)

    #if beforeImg is None:
    #    raise NameError("Can't find image: \"" + name1 + '\"')
    #elif afterImg is None:
    #    raise NameError("Can't find image: \"" + name2 + '\"')

    #beforeImg = cv2.imread(os.path.join(path, name1), cv2.IMREAD_GRAYSCALE).astype(float)
    #afterImg = cv2.imread(os.path.join(path, name2), cv2.IMREAD_GRAYSCALE).astype(float)

    #removing noise
    #beforeImg  = cv2.GaussianBlur(beforeImg, (5, 5), 0)
    #afterImg = cv2.GaussianBlur(afterImg, (5, 5), 0)

    beforeImg = np.array(copy.deepcopy(name1))
    afterImg = np.array(copy.deepcopy(name2))

    # set up initial values
    u = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    v = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    fx, fy, ft = get_derivatives(beforeImg, afterImg)
    avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                            [1 / 6, 0, 1 / 6],
 [1  /12, 1  /6, 1  /12]], float)
    iter_counter = 0
    while True:
        iter_counter += 1
        u_avg = filter2(u, avg_kernel)
        v_avg = filter2(v, avg_kernel)
        p = fx * u_avg + fy * v_avg + ft
        d = 4 * alpha**2 + fx**2 + fy**2
        prev = u

        u = u_avg - fx * (p / d)
        v = v_avg - fy * (p / d)

        diff = np.linalg.norm(u - prev, 2)
        print('norm = ', diff, '\t k = ', iter_counter)
        #converges check (at most 300 iterations)
        if  diff < delta or iter_counter > 300:
            # print("iteration number: ", iter_counter)
            break

    k = 0
    #draw_quiver(u, v, beforeImg)

    return [u, v]

tic = time.perf_counter()

u1,v1 = computeHS(A2, A1, alpha = 1, delta = 10**-6)

time2 = time.perf_counter() - tic

u1 = u1.tolist()
v1 = v1.tolist()

ax = plt.figure().gca()
ax.imshow(A3, cmap = 'gray')

for i in range(0, size[0] + 2*frame_width, step):
    for j in range(0, size[1] + 2*frame_width, step):
        ax.arrow(i, j, step*u1[j][i], step*v1[j][i], length_includes_head = True, head_width = 0.3*step, overhang = 1.5, head_length = 0.3*step, width = 0.01)

plt.draw()
plt.show()


ax = plt.figure().gca()
ax.imshow(A3, cmap = 'gray')

for i in range(0, size[0] + 2*frame_width, step):
    for j in range(0, size[1] + 2*frame_width, step):
        c = math.sqrt(u1[j][i]**2+v1[j][i]**2)
        if c != 0:
            ax.arrow(i, j, step*u1[j][i]/c, step*v1[j][i]/c, length_includes_head = True, head_width = 0.15*step, overhang = 1, head_length = 0.2*step, width = 0.01)
        else:
            ax.arrow(i, j, step*u1[j][i], step*v1[j][i], length_includes_head = True, head_width = 0.15*step, overhang = 1, head_length = 0.2*step, width = 0.01)

plt.draw()
plt.show()

#--------------------для готового изображения закомметировать------------------
for j in range(frame_width, size[0]+frame_width):
    for i in range(frame_width, size[1]+frame_width):
        tan[i][j] = abs(-1-v1[i][j]/u1[i][j])

frame_w = []
for t in range(0, size[0]):
    sum = 0
    for j in range(frame_width+t, size[0]+frame_width-t):
        for i in range(frame_width+t, size[1]+frame_width-t):
            sum += tan[i][j]
    accur1.append(sum/((size[0]-t)*(size[1]-t)))
    frame_w.append(t)
#-------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------------------
print('my metod time: ', time1)
print('other metod time: ', time2)


#--------------------для готового изображения закомметировать------------------
fig, ax = plt.subplots()

plt.plot(frame_w, accur0, color='r', label='реализация 1')
plt.plot(frame_w, accur1, color='b', label='реализация 2')
plt.xlabel('Расстояние до границы')
plt.legend()

plt.show()
#-----------------------------------------------------------------------
