# Part 3: Single-View Geometry

## Usage
This code snippet provides an overall code structure and some interactive plot interfaces for the *Single-View Geometry* section of Assignment 3. In [main function](#Main-function), we outline the required functionalities step by step. Some of the functions which involves interactive plots are already provided, but [the rest](#Your-implementation) are left for you to implement.

## Package installation
- In this code, we use `tkinter` package. Installation instruction can be found [here](https://anaconda.org/anaconda/tk).

# Common imports


```python
# %matplotlib tk
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
```


```python
plt.rcParams["figure.figsize"] = [18,12]
```

# Provided functions


```python
def get_input_lines(im, min_lines=3):
    """
    Allows user to input line segments; computes centers and directions.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        min_lines: minimum number of lines required
    Returns:
        n: number of lines from input
        lines: np.ndarray of shape (3, n)
            where each column denotes the parameters of the line equation
        centers: np.ndarray of shape (3, n)
            where each column denotes the homogeneous coordinates of the centers
    """
    n = 0
    lines = np.zeros((3, 0))
    centers = np.zeros((3, 0))

    plt.figure()
    plt.imshow(im)
    plt.show()
    print('Set at least %d lines to compute vanishing point' % min_lines)
    while True:
        print('Click the two endpoints, use the right key to undo, and use the middle key to stop input')
        clicked = plt.ginput(2, timeout=0, show_clicks=True)
        if not clicked or len(clicked) < 2:
            if n < min_lines:
                print('Need at least %d lines, you have %d now' % (min_lines, n))
                continue
            else:
                # Stop getting lines if number of lines is enough
                break

        # Unpack user inputs and save as homogeneous coordinates
        pt1 = np.array([clicked[0][0], clicked[0][1], 1])
        pt2 = np.array([clicked[1][0], clicked[1][1], 1])
        # Get line equation using cross product
        # Line equation: line[0] * x + line[1] * y + line[2] = 0
        line = np.cross(pt1, pt2)
        lines = np.append(lines, line.reshape((3, 1)), axis=1)
        # Get center coordinate of the line segment
        center = (pt1 + pt2) / 2
        centers = np.append(centers, center.reshape((3, 1)), axis=1)

        # Plot line segment
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='b')

        n += 1

    return n, lines, centers
```


```python
def plot_lines_and_vp(im, lines, vp):
    """
    Plots user-input lines and the calculated vanishing point.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        lines: np.ndarray of shape (3, n)
            where each column denotes the parameters of the line equation
        vp: np.ndarray of shape (3, )
    """
    bx1 = min(1, vp[0] / vp[2]) - 10
    bx2 = max(im.shape[1], vp[0] / vp[2]) + 10
    by1 = min(1, vp[1] / vp[2]) - 10
    by2 = max(im.shape[0], vp[1] / vp[2]) + 10

    plt.figure()
    plt.imshow(im)
    for i in range(lines.shape[1]):
        if lines[0, i] < lines[1, i]:
            pt1 = np.cross(np.array([1, 0, -bx1]), lines[:, i])
            pt2 = np.cross(np.array([1, 0, -bx2]), lines[:, i])
        else:
            pt1 = np.cross(np.array([0, 1, -by1]), lines[:, i])
            pt2 = np.cross(np.array([0, 1, -by2]), lines[:, i])
        pt1 = pt1 / pt1[2]
        pt2 = pt2 / pt2[2]
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g')

    plt.plot(vp[0] / vp[2], vp[1] / vp[2], 'ro')
    plt.show()
```


```python
def get_top_and_bottom_coordinates(im, obj):
    """
    For a specific object, prompts user to record the top coordinate and the bottom coordinate in the image.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        obj: string, object name
    Returns:
        coord: np.ndarray of shape (3, 2)
            where coord[:, 0] is the homogeneous coordinate of the top of the object and coord[:, 1] is the homogeneous
            coordinate of the bottom
    """
    plt.figure()
    plt.imshow(im)

    print('Click on the top coordinate of %s' % obj)
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x1, y1 = clicked[0]
    # Uncomment this line to enable a vertical line to help align the two coordinates
    # plt.plot([x1, x1], [0, im.shape[0]], 'b')
    print('Click on the bottom coordinate of %s' % obj)
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x2, y2 = clicked[0]

    plt.plot([x1, x2], [y1, y2], 'b')

    return np.array([[x1, x2], [y1, y2], [1, 1]])
```

# Your implementation


```python
def to_homo(a):
    """
    convert (n x dim) to (n x dim + 1)
    """
    return np.r_[a.T, np.ones((1,) + a.T.shape[1:])].T
def from_homo(a):
    """
    convert (n x dim + 1) to (n x dim)
    """
    return (a.T[:-1] / a.T[[-1]]).T
```


```python
def get_vanishing_point(lines):
    """
    Solves for the vanishing point using the user-input lines.
    """
    # transpose so 1 line on each row
    # then intersect each line with the "next" line, wrapping around
    pts = to_homo(from_homo(np.cross(lines.T, lines.T[[1,2,0]])))
    return np.mean(pts,axis=0)
```


```python
def get_horizon_line(vpts):
    """
    Calculates the ground horizon line.
    """
    assert vpts.shape == (3,2)
    # line from left to right vpt
    horizon_line = np.cross(vpts[:,0], vpts[:,1])
    # normalize line so a^2 + b^2 = 1
    lam = np.sqrt(1 / np.sum(horizon_line[:2] ** 2))
    return horizon_line * lam
```


```python
def plot_horizon_line(im, horizon):
    """
    Plots the horizon line.
    """
    # left and right side of the image
    h,w = im.shape[:2]
    vert_edges = [[1,0,0],[1,0,-w]]
    # intersect the sides with horizon line
    pts = from_homo(np.cross(vert_edges, horizon[None,:]))
    
    plt.figure()
    plt.imshow(im)
    plt.plot(pts[:,0],pts[:,1])
    plt.show()
#     plt.close()
```


```python
from sympy import symbols
from sympy.matrices import Matrix
from sympy.solvers import solve

def K_mat(f,px,py):
    return [[f,0,px],
            [0,f,py],
            [0,0,1]]

def get_camera_parameters(vpts):
    """
    Computes the camera parameters. Hint: The SymPy package is suitable for this.
    """
    assert vpts.shape == (3,3)
    f,px,py = symbols("f,px,py")
    K = Matrix(K_mat(f,px,py))
    # vpt in 3d should be orthogonal to next vpt
    eqs = [ vpts[:,[i]].T * K.inv().T * K.inv() * vpts[:,[(i+1) % 3]] for i in range(3) ]
    f,px,py = solve(eqs,f,px,py)[0]
    return f,px,py
```


```python
def get_rotation_matrix(f,px,py,vpts):
    """
    Computes the rotation matrix using the camera parameters.
    """
    K = np.array(K_mat(f,px,py), dtype=float)
    
    vpts_right_middle_left = vpts[:,np.argsort(vpts[0,:])[::-1]]
    
    R = np.linalg.inv(K) @ vpts_right_middle_left
    
    return R / np.linalg.norm(R,axis=0)
```


```python
def estimate_height(reference_coords,object_coords,horizon,vertical_vpt,ax):
    """
    Estimates height for a specific object using the recorded coordinates. You might need to plot additional images here for
    your report.
    """
    # <YOUR IMPLEMENTATION>
    
    r, b = reference_coords.T
    t_o, b_o = object_coords.T
    
    
    # vanishing point on horizon of line through b_o and b (bottoms) 
    v = to_homo(from_homo(np.cross(np.cross(b, b_o),horizon)))
    t = to_homo(from_homo(np.cross(np.cross(v, t_o), np.cross(b,r))))
    
    pts = np.r_[reference_coords.T, object_coords.T, v[None,:], t[None,:]]
    ax.plot(pts[:,0],pts[:,1],"r+")
    ax.plot(pts[[0,1],0],pts[[0,1],1],"c-.")
    ax.plot(pts[[0,5],0],pts[[0,5],1],"y-.")
    
    ax.plot(pts[[1,3],0],pts[[1,3],1],"r-.")
    ax.plot(pts[[2,5],0],pts[[2,5],1],"r-.")

    ax.plot(pts[[3,4,4,2],0],pts[[3,4,4,2],1],"g-.")

    
    for coord,label in zip(pts[:,:2],["r","b","t_o","b_o","v","t"]):
        ax.annotate(label,coord,color='r',size=14)
    
    ratio = (np.linalg.norm(t - b) / np.linalg.norm(r - b)) * (np.linalg.norm(vertical_vpt - r) / np.linalg.norm(vertical_vpt - t))
    
    return ratio
```

# Main function


```python
!mkdir -p "user_input"
def np_cache(fname, func, load_cache=np.loadtxt, write_cache=np.savetxt):
    try:
        result = load_cache(fname)
    except IOError:
        result = func()
        write_cache(fname, result)
    return result
```


```python
im = np.asarray(Image.open('CSL.jpeg'))

# Part 1
# Get vanishing points for each of the directions
num_vpts = 3
vpts = np.zeros((3, num_vpts))
for i in range(num_vpts):
    print('Getting vanishing point %d' % i)
    # Get at least three lines from user input
    lines = np_cache(f"user_input/lines_dir_{i}.txt", lambda: get_input_lines(im, i)[1])
    
    # <YOUR IMPLEMENTATION> Solve for vanishing point
    vpts[:, i] = get_vanishing_point(lines)
    # Plot the lines and the vanishing point
    plot_lines_and_vp(im, lines, vpts[:, i])
    print ("Vanishing Point:", vpts[:,i][:2])

# <YOUR IMPLEMENTATION> Get the ground horizon line
horizon_line = get_horizon_line(vpts[:,:2])
# <YOUR IMPLEMENTATION> Plot the ground horizon line
plot_horizon_line(im, horizon_line)
print (f"Horizon Line:", horizon_line)

# Part 2
# <YOUR IMPLEMENTATION> Solve for the camera parameters (f, u, v)
f, u, v = get_camera_parameters(vpts)
print ("Focal Length:", f)
print (f"PX: {u:.03f} PY: {v:.03f}")

# Part 3
# <YOUR IMPLEMENTATION> Solve for the rotation matrix
R = get_rotation_matrix(f,u,v,vpts)
print ("Rotation Matrix:\n", R)
tmp = np.array(K_mat(f,u,v), dtype=float) @ R
# print (tmp / tmp[[-1]])

# Part 4
# Record image coordinates for each object and store in map
objects = ('person', 'CSL building', 'the spike statue', 'the lamp posts')
coords = dict()
for obj in objects:
    coords[obj] = np_cache(f"user_input/{obj}_coords.txt", lambda: get_top_and_bottom_coordinates(im, obj))

vertical_vpt = vpts[:,np.argmax(vpts[1,:])]
meter_per_inch = 0.0254
person_5_6_m = (5 * 12 + 6) * meter_per_inch # 5 foot 6 in person
person_6_m = (6 * 12) * meter_per_inch # 5 foot 6 in person
    
# <YOUR IMPLEMENTATION> Estimate heights
for obj in objects[1:]:
    fig, ax = plt.subplots()
    ax.imshow(im, interpolation='none')
#     print('Estimating height of %s' % obj)
    height_ratio = estimate_height(coords['person'], coords[obj], horizon_line, vertical_vpt, ax)
    obj_height = height_ratio * person_5_6_m 
    plt.show()
    print (f"{obj} height (person is 5'6\" [{person_5_6_m:.03f}m])= {height_ratio * person_5_6_m:.03f}m")
    print (f"{obj} height (person is 6' [{person_6_m:.03f}m])= {height_ratio * person_6_m:.03f}m")


```

    Getting vanishing point 0



    
![png](Part3_files/Part3_18_1.png)
    


    Vanishing Point: [-272.07231662  213.70742266]
    Getting vanishing point 1



    
![png](Part3_files/Part3_18_3.png)
    


    Vanishing Point: [1389.39566473  232.75706159]
    Getting vanishing point 2



    
![png](Part3_files/Part3_18_5.png)
    


    Vanishing Point: [ 499.29296913 6929.9740703 ]



    
![png](Part3_files/Part3_18_7.png)
    


    Horizon Line: [-1.14647933e-02  9.99934277e-01 -2.16812630e+02]
    Focal Length: -824.209705045857
    PX: 575.008 PY: 326.290
    Rotation Matrix:
     [[-0.70057686  0.01137654  0.71348626]
     [ 0.08046147 -0.99223678  0.09482686]
     [ 0.70902611  0.12384165  0.69422275]]



    
![png](Part3_files/Part3_18_9.png)
    


    CSL building height (person is 5'6" [1.676m])= 18.729m
    CSL building height (person is 6' [1.829m])= 20.432m



    
![png](Part3_files/Part3_18_11.png)
    


    the spike statue height (person is 5'6" [1.676m])= 11.636m
    the spike statue height (person is 6' [1.829m])= 12.694m



    
![png](Part3_files/Part3_18_13.png)
    


    the lamp posts height (person is 5'6" [1.676m])= 4.738m
    the lamp posts height (person is 6' [1.829m])= 5.169m


## Extra Credit


```python
extra_objects = ('person_white', 'person_black', 'person_red', 'window_top_to_ground', 'window_bottom_to_ground')
heights = dict()

for obj in extra_objects:
    coords[obj] = np_cache(f"user_input/{obj}_coords.txt", lambda: get_top_and_bottom_coordinates(im, obj))

for obj in extra_objects:
    fig, ax = plt.subplots()
    ax.imshow(im, interpolation='none')
    height_ratio = estimate_height(coords['person'], coords[obj], horizon_line, vertical_vpt, ax)
    plt.show()
    heights[obj] = height_ratio * person_5_6_m
    print (f"{obj} height (person is 5'6\" [{person_5_6_m:.03f}m])= {heights[obj]:.03f}m")
#     print (f"{obj} height (person is 6' [{person_6_m:.03f}m])= {height_ratio * person_6_m:.03f}m")
    
window_height = heights['window_top_to_ground'] - heights['window_bottom_to_ground']
print (f"Window Height={window_height:.03f}m")
```


    
![png](Part3_files/Part3_20_0.png)
    


    person_white height (person is 5'6" [1.676m])= 1.668m



    
![png](Part3_files/Part3_20_2.png)
    


    person_black height (person is 5'6" [1.676m])= 1.572m



    
![png](Part3_files/Part3_20_4.png)
    


    person_red height (person is 5'6" [1.676m])= 1.737m



    
![png](Part3_files/Part3_20_6.png)
    


    window_top_to_ground height (person is 5'6" [1.676m])= 12.954m



    
![png](Part3_files/Part3_20_8.png)
    


    window_bottom_to_ground height (person is 5'6" [1.676m])= 10.467m
    Window Height=2.486m



```python

```


```python

```
