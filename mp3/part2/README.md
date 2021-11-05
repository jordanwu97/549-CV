# Part 2: Fundamental Matrix Estimation, Camera Calibration, Triangulation


```python
!pip install -q -U kaleido plotly==5.3.1
```

    [K     |████████████████████████████████| 79.9 MB 46 kB/s 
    [K     |████████████████████████████████| 23.9 MB 14 kB/s 
    [?25h


```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, edgeitems=12, linewidth=200)
import plotly.express as px
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D

```


```python
def to_homo(x):
    return np.c_[x,np.broadcast_to(1,len(x))]

def from_homo(x):
    return x[:,:-1] / x[:,[-1]]
```

## Fundamental Matrix Estimation


```python
prefix = "/content/drive/MyDrive/549/mp3/"

I1 = Image.open(prefix + 'MP3_part2_data/library1.jpg');
I2 = Image.open(prefix + 'MP3_part2_data/library2.jpg');
matches = np.loadtxt(prefix + 'MP3_part2_data/library_matches.txt'); 

def fit_fundamental_normalize(matches):

    def get_T(x):
        m_x,m_y = -1 * x.mean(axis=0)
        s_x,s_y = np.sqrt(2) / x.std(axis=0)

        scaling = np.array([[s_x,0,0],
                   [0,s_y,0],
                   [0,0,1]])

        shift = np.array([[1,0,m_x],
                          [0,1,m_y],
                          [0,0,1]])
    
        return scaling @ shift
    
    left = matches[:,:2]
    right = matches[:,2:]

    T_left = get_T(left)
    T_right = get_T(right)

    left_norm = from_homo(to_homo(left) @ T_left.T)
    right_norm = from_homo(to_homo(right) @ T_right.T)

    F_norm = fit_fundamental(np.c_[left_norm,right_norm])

    return T_right.T @ F_norm @ T_left

def fit_fundamental(matches):
    left = matches[:,:2]
    right = matches[:,2:]

    U_left = np.ones((len(matches), 9))
    U_left[:,[0,1,3,4,6,7]] = left[:,[0,1,0,1,0,1]]

    U_right = np.ones((len(matches), 9))
    U_right[:, [0,1,2,3,4,5]] = right[:,[0,0,0,1,1,1]]

    U = U_left * U_right

    # find eigenvalue of U.T @ U
    eigval,eigvec = np.linalg.eig(U.T @ U)
    F_init = eigvec[:,np.argmin(eigval)]

    # enforcing rank two
    u,s,v = np.linalg.svd(F_init.reshape(3,3))
    s[-1] = 0
    F = u @ np.diag(s) @ v

    return F
###

def get_closest(F, x, x_prime, endpoints_len, ax=None):
    # to homogenous
    N = len(x)
    M = np.c_[x, np.ones((N,1))].transpose() 
    L1 = np.matmul(F, M).transpose() # transform points from 
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to x_prime
    l = np.sqrt(L1[:,0]**2 + L1[:,1]**2)
    L = np.divide(L1,np.kron(np.ones((3,1)),l).transpose())# rescale the line
    pt_line_dist = np.multiply(L, np.c_[x_prime, np.ones((N,1))]).sum(axis = 1)
    closest_pt = x_prime - np.multiply(L[:,0:2],np.kron(np.ones((2,1)), pt_line_dist).transpose())

    if ax != None:
        # find endpoints of segment on epipolar line (for display purposes)
        pt1 = closest_pt - np.c_[L[:,1], -L[:,0]]*endpoints_len# offset from the closest point is 10 pixels
        pt2 = closest_pt + np.c_[L[:,1], -L[:,0]]*endpoints_len

        # display points and segments of corresponding epipolar lines
        # fig, ax = plt.subplots(figsize=(20,20))
        # ax.set_aspect('equal')
        # ax.imshow(np.array(img).astype(float) / 255)
        ax.plot(x_prime[:,0], x_prime[:,1],  '+r')
        ax.plot([x_prime[:,0], closest_pt[:,0]],[x_prime[:,1], closest_pt[:,1]], 'b')
        ax.plot([pt1[:,0], pt2[:,0]],[pt1[:,1], pt2[:,1]], 'g')

    return closest_pt

def run(matches, F_func, imgs, endpoints_len, figsize):

    F = F_func(matches)

    fig, axs = plt.subplots(1,2,figsize=figsize)
    fig.suptitle(f"{F_func.__name__}")
    for ax in axs:
        ax.axis('off')
        ax.set_aspect('equal')
    
    axs[0].imshow(np.array(imgs[0]).astype(float) / 255)
    axs[1].imshow(np.array(imgs[1]).astype(float) / 255)

    im0_closest = get_closest(F.T, matches[:,2:], matches[:,:2], endpoints_len, axs[0])
    im1_closest = get_closest(F,   matches[:,:2], matches[:,2:], endpoints_len, axs[1])

    h,w = np.array(imgs[0]).shape[:2]
    for ax in axs:
        ax.set_xlim([0,w])
        ax.set_ylim([h,0])
    
    fig.tight_layout()

    residuals = np.linalg.norm(np.r_[
        matches[:,:2] - im0_closest,
        matches[:,2:] - im1_closest
    ], axis=1)

    plt.show()

    print (f"residual({F_func.__name__}):{residuals.mean():.06f}")

figsize=(20,9)
run(matches, fit_fundamental, [I1,I2], 10, figsize)
run(matches, fit_fundamental_normalize, [I1,I2], 10, figsize)
```


    
![png](Part2_files/Part2_5_0.png)
    


    residual(fit_fundamental):0.325625



    
![png](Part2_files/Part2_5_2.png)
    


    residual(fit_fundamental_normalize):0.179055



```python
prefix = "/content/drive/MyDrive/549/mp3/"

I1 = Image.open(prefix + 'MP3_part2_data/lab1.jpg');
I2 = Image.open(prefix + 'MP3_part2_data/lab2.jpg');
matches = np.loadtxt(prefix + 'MP3_part2_data/lab_matches.txt'); 
figsize=(20,8)
run(matches, fit_fundamental, [I1,I2], 20, figsize)
run(matches, fit_fundamental_normalize, [I1,I2], 20, figsize)
```


    
![png](Part2_files/Part2_6_0.png)
    


    residual(fit_fundamental):2.423569



    
![png](Part2_files/Part2_6_2.png)
    


    residual(fit_fundamental_normalize):0.633007


## Camera Calibration


```python
x_match = np.loadtxt(prefix+"MP3_part2_data/lab_matches.txt")
X = np.loadtxt(prefix+"MP3_part2_data/lab_3d.txt")

def getA(x,X):
    X_homo = np.c_[X,np.ones((len(X),1))]
    A = np.zeros((2 * len(X), 12))
    A[::2,  [0,1,2,3,8,9,10,11]] = X_homo[:,[0,1,2,3,0,1,2,3]]
    A[::2,  [8,9,10,11]] = A[::2, [8,9,10,11]] * x[:,[0]] * -1
    A[1::2, [4,5,6,7,8,9,10,11]] = X_homo[:,[0,1,2,3,0,1,2,3]]
    A[1::2, [8,9,10,11]] = A[1::2,[8,9,10,11]] * x[:,[1]] * -1
    return A

def getP(x,X):
    A = getA(x, X)
    eigval, eigvec = np.linalg.eig(A.T @ A)
    p = eigvec[:,np.argmin(eigval)].reshape(3,4)
    return p
```


```python
def evaluate_points(M, points_2d, points_3d):
    """
    Visualize the actual 2D points and the projected 2D points calculated from
    the projection matrix
    You do not need to modify anything in this function, although you can if you
    want to
    :param M: projection matrix 3 x 4
    :param points_2d: 2D points N x 2
    :param points_3d: 3D points N x 3
    :return:
    """
    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(M, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual

P_left = getP(x_match[:,:2], X)
P_right = getP(x_match[:,2:], X)

print (f"P(lab1)\n", P_left)
print (f"P(lab2)\n", P_right)
print (f"residuals(lab1): {evaluate_points(P_left, x_match[:,:2], X)[1]:.06f}")
print (f"residuals(lab2): {evaluate_points(P_right, x_match[:,2:], X)[1]:.06f}")
```

    P(lab1)
     [[ 3.100e-03  1.462e-04 -4.485e-04 -9.789e-01]
     [ 3.070e-04  6.372e-04 -2.774e-03 -2.041e-01]
     [ 1.679e-06  2.748e-06 -6.840e-07 -1.329e-03]]
    P(lab2)
     [[ 6.932e-03 -4.017e-03 -1.326e-03 -8.267e-01]
     [ 1.548e-03  1.025e-03 -7.274e-03 -5.625e-01]
     [ 7.609e-06  3.710e-06 -1.902e-06 -3.388e-03]]
    residuals(lab1): 13.545700
    residuals(lab2): 15.544956


## Camera Centers


```python
def find_camera_center(P):
    C = scipy.linalg.null_space(P).T
    C = C / C[:,[-1]]
    return C

x_match = np.loadtxt(prefix+"MP3_part2_data/lab_matches.txt")
X = np.loadtxt(prefix+"MP3_part2_data/lab_3d.txt")

Ps = [
    getP(x_match[:,:2], X),
    getP(x_match[:,2:], X),
    np.loadtxt(prefix + "MP3_part2_data/library1_camera.txt"),
    np.loadtxt(prefix + "MP3_part2_data/library2_camera.txt")
]

names = ["lab1","lab2","library1","library2"]

_ = [ print (f"{name}: {find_camera_center(p)}") for name, p in zip(names, Ps) ]
```

    lab1: [[305.833 304.201  30.137   1.   ]]
    lab2: [[303.1   307.184  30.422   1.   ]]
    library1: [[  7.289 -21.521  17.735   1.   ]]
    library2: [[  6.894 -15.392  23.415   1.   ]]


## Triangulation


```python

def create_3D(matches,P_left,P_right):

    n = len(matches)

    # create cross product matrices from points
    cross_mat = lambda x_homo: np.cross(x_homo[:,None,:], (np.identity(3) * -1)[None,:,:])
    left_cross = cross_mat(to_homo(matches[:,:2]))
    right_cross = cross_mat(to_homo(matches[:,2:]))

    # matmul to get x cross P
    left_cross_P = np.einsum("abc,cd", left_cross, P_left)
    right_cross_P = np.einsum("abc,cd", right_cross, P_right)

    # setup systems of equation, copying 2 equations from left, 2 from right
    U = np.zeros((n,4,4))
    U[:,:2,:] = left_cross_P[:,:2,:]
    U[:,2:,:] = right_cross_P[:,:2,:]

    _,_,v = np.linalg.svd(U)
    return from_homo(v[:,-1,:])

def run_3D(matches,P_left,P_right,X_true=None):
    X_predict = create_3D(matches, P_left, P_right)
    
    C_left = from_homo(find_camera_center(P_left))
    C_right = from_homo(find_camera_center(P_right))

    print (C_left)
    print (C_right)

    if X_true is not None:
        print ("residuals:", np.linalg.norm(X_true - X_predict, axis=1).mean())

    xs,ys,zs = np.c_[X_predict.T, C_left.T, C_right.T]
    color = (["points"] * len(X_predict)) + ["camera_left","camera_right"]
    fig = px.scatter_3d(x=xs,y=ys,z=zs,color=color,symbol=color)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig
```


```python
x_match = np.loadtxt(prefix+"MP3_part2_data/lab_matches.txt")
X = np.loadtxt(prefix+"MP3_part2_data/lab_3d.txt")

# lab pic 1 is from the right, pic 2 if from the left
P_right = getP(x_match[:,:2], X)
P_left = getP(x_match[:,2:], X)
fig = run_3D(x_match[:,[2,3,0,1]], P_left, P_right, X_true=np.loadtxt(prefix+"MP3_part2_data/lab_3d.txt"))
```

    [[303.1   307.184  30.422]]
    [[305.833 304.201  30.137]]
    residuals: 0.013320683982140702



```python
!mkdir -p {prefix}out
fig.update_layout(scene_camera=dict(eye=dict(x=-2, y=-0.5, z=0.85)))
fig.write_image(prefix + "out/lab_out_3d_l.jpg")

fig.update_layout(scene_camera=dict(eye=dict(x=-0.5, y=-2, z=0.85)))
fig.write_image(prefix + "out/lab_out_3d_r.jpg")
fig.write_html(prefix + "out/lab_out_3d.html")
```


```python
Image.open(prefix+"out/lab_out_3d_l.jpg")
```




    
![png](Part2_files/Part2_16_0.png)
    




```python
Image.open(prefix+"out/lab_out_3d_r.jpg")
```




    
![png](Part2_files/Part2_17_0.png)
    








```python
P_left = np.loadtxt(prefix + "MP3_part2_data/library1_camera.txt")
P_right = np.loadtxt(prefix + "MP3_part2_data/library2_camera.txt")
x_match = np.loadtxt(prefix + "MP3_part2_data/library_matches.txt")

fig = run_3D(x_match, P_left, P_right)
```

    [[  7.289 -21.521  17.735]]
    [[  6.894 -15.392  23.415]]



```python
camera = dict(
    up=dict(x=-1, y=0, z=0),
    eye=dict(x=-0.25, y=-1.5, z=2)
)
fig.update_layout(scene_camera=camera)
fig.write_image(prefix + "out/library_out_3d_r.jpg")

fig.update_layout(scene_camera=dict(eye=dict(x=-0.25, y=-2, z=0.25)))
fig.write_image(prefix + "out/library_out_3d_l.jpg")
fig.write_html(prefix + "out/library_out_3d.html")
```


```python
Image.open(prefix+"out/library_out_3d_l.jpg")
```




    
![png](Part2_files/Part2_21_0.png)
    




```python
Image.open(prefix+"out/library_out_3d_r.jpg")
```




    
![png](Part2_files/Part2_22_0.png)
    




```python

```
