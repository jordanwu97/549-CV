# Common imports


```python
%matplotlib inline
import os
import sys
import glob
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import time
```


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).


# Provided functions
### Image loading and saving


```python
def LoadFaceImages(pathname, subject_name, num_images):
    """
    Load the set of face images.  
    The routine returns
        ambimage: image illuminated under the ambient lighting
        imarray: a 3-D array of images, h x w x Nimages
        lightdirs: Nimages x 3 array of light source directions
    """

    def load_image(fname):
        return np.asarray(Image.open(fname))

    def fname_to_ang(fname):
        yale_name = os.path.basename(fname)
        return int(yale_name[12:16]), int(yale_name[17:20])

    def sph2cart(az, el, r):
        rcos_theta = r * np.cos(el)
        x = rcos_theta * np.cos(az)
        y = rcos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    ambimage = load_image(
        os.path.join(pathname, subject_name + '_P00_Ambient.pgm'))
    im_list = glob.glob(os.path.join(pathname, subject_name + '_P00A*.pgm'))
    if num_images <= len(im_list):
        im_sub_list = np.random.choice(im_list, num_images, replace=False)
    else:
        print(
            'Total available images is less than specified.\nProceeding with %d images.\n'
            % len(im_list))
        im_sub_list = im_list
    im_sub_list.sort()
    imarray = np.stack([load_image(fname) for fname in im_sub_list], axis=-1)
    Ang = np.array([fname_to_ang(fname) for fname in im_sub_list])

    x, y, z = sph2cart(Ang[:, 0] / 180.0 * np.pi, Ang[:, 1] / 180.0 * np.pi, 1)
    lightdirs = np.stack([y, z, x], axis=-1)
    return ambimage, imarray, lightdirs
```


```python
def save_outputs(subject_name, albedo_image, surface_normals):
    im = Image.fromarray((albedo_image*255).astype(np.uint8))
    im.save("%s_albedo.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,0]*128+128).astype(np.uint8))
    im.save("%s_normals_x.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,1]*128+128).astype(np.uint8))
    im.save("%s_normals_y.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,2]*128+128).astype(np.uint8))
    im.save("%s_normals_z.jpg" % subject_name)
```

### Plot the height map


```python
def set_aspect_equal_3d(ax):
    """https://stackoverflow.com/questions/13685386"""
    """Fix equal aspect bug for 3D plots."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)
    plot_radius = max([
        abs(lim - mean_)
        for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
        for lim in lims
    ])
    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def display_output(albedo_image, height_map):
    fig = plt.figure()
    plt.imshow(albedo_image, cmap='gray')
    plt.axis('off')
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.view_init(20, 20)
    X = np.arange(albedo_image.shape[0])
    Y = np.arange(albedo_image.shape[1])
    X, Y = np.meshgrid(Y, X)
    H = np.flipud(np.fliplr(height_map))
    A = np.flipud(np.fliplr(albedo_image))
    A = np.stack([A, A, A], axis=-1)
    ax.xaxis.set_ticks([])
    ax.xaxis.set_label_text('Z')
    ax.yaxis.set_ticks([])
    ax.yaxis.set_label_text('X')
    ax.zaxis.set_ticks([])
    ax.yaxis.set_label_text('Y')
    surf = ax.plot_surface(
        H, X, Y, cmap='gray', facecolors=A, linewidth=0, antialiased=False)
    set_aspect_equal_3d(ax)
```


```python
def display_albedo(albedo_image, name):
    fig = plt.figure()
    plt.imshow(albedo_image, cmap='gray')
    plt.axis('off')
    plt.title(name)

def display_height_3d(albedo_image, height_map, name):
    fig = plt.figure(figsize = (21,7))
    plt.suptitle(name)

    for idx, angle in enumerate([20,60,90]):
        ax = plt.subplot(1, 3, idx+1, projection='3d')
        ax.view_init(20, angle)
        X = np.arange(albedo_image.shape[0])
        Y = np.arange(albedo_image.shape[1])
        X, Y = np.meshgrid(Y, X)
        H = np.flipud(np.fliplr(height_map))
        A = np.flipud(np.fliplr(albedo_image))
        A = np.stack([A, A, A], axis=-1)
        ax.xaxis.set_ticks([])
        ax.xaxis.set_label_text('Z')
        ax.yaxis.set_ticks([])
        ax.yaxis.set_label_text('X')
        ax.zaxis.set_ticks([])
        ax.yaxis.set_label_text('Y')
        surf = ax.plot_surface(
            H, X, Y, cmap='gray', facecolors=A, linewidth=0, antialiased=False)
        set_aspect_equal_3d(ax)
        plt.title(f"angle={angle}")
```

### Plot the surface norms. 


```python
def plot_surface_normals(surface_normals, name):
    """
    surface_normals: h x w x 3 matrix.
    """
    fig = plt.figure(name)
    ax = plt.subplot(1, 3, 1)
    ax.axis('off')
    ax.set_title('X')
    im = ax.imshow(surface_normals[:,:,0])
    ax = plt.subplot(1, 3, 2)
    ax.axis('off')
    ax.set_title('Y')
    im = ax.imshow(surface_normals[:,:,1])
    ax = plt.subplot(1, 3, 3)
    ax.axis('off')
    ax.set_title('Z')
    im = ax.imshow(surface_normals[:,:,2])
    plt.suptitle(name)
```

# Your implementation


```python
def preprocess(ambimage, imarray):
    """
    preprocess the data: 
        1. subtract ambient_image from each image in imarray.
        2. make sure no pixel is less than zero.
        3. rescale values in imarray to be between 0 and 1.
    Inputs:
        ambimage: h x w
        imarray: h x w x Nimages
    Outputs:
        processed_imarray: h x w x Nimages
    """

    subtract = imarray - ambimage[:,:,None]
    clipped = np.maximum(subtract, 0)
    scaled = clipped / 255

    processed_imarray = scaled

    return processed_imarray
```


```python
def photometric_stereo(imarray, light_dirs):
    """
    Inputs:
        imarray:  h x w x Nimages
        light_dirs: Nimages x 3, unit vectors
    Outputs:
        albedo_image: h x w
        surface_norms: h x w x 3
    """

    # pxs = h*w (total number of pixels per image, flattened in row major order)
    h,w,Nimages = imarray.shape
    pxs = h*w

    # transpose imarray to become (Nimages * h * w), then reshape (Nimages x pxs)
    im_flattened = imarray.transpose((2,0,1)).reshape(Nimages,pxs)

    # fit G
    # im_flattened (Nimages x pxs)  = light_dirs (Nimages x 3) @ G (3 x pxs)
    G = np.linalg.lstsq(light_dirs, im_flattened, rcond=None)[0]

    # G reshaped to h x w x 3
    G_img = G.transpose().reshape(h,w,3)

    albedo_image = np.linalg.norm(G_img,axis=2)
    surface_normals = G_img / albedo_image[:,:,None]

    return albedo_image, surface_normals
```


```python
from numpy.core.defchararray import rpartition
def get_surface(surface_normals, integration_method):
    """
    Inputs:
        surface_normals:h x w x 3
        integration_method: string in ['average', 'column', 'row', 'random']
    Outputs:
        height_map: h x w
    """

    # partial derivatives
    f_x = surface_normals[:,:,0] / surface_normals[:,:,2]
    f_y = surface_normals[:,:,1] / surface_normals[:,:,2]

    # add across row
    x_cumsum = np.cumsum(f_x, axis=1)
    # add down column
    y_cumsum = np.cumsum(f_y, axis=0)

    # go across the first row, then down the columns
    height_map_row_first = x_cumsum[0,:][None,:] + y_cumsum

    if integration_method == "row":
        return height_map_row_first

    # go down the columns, then across the rows
    height_map_column_first = y_cumsum[:,0][:,None] + x_cumsum

    if integration_method == "column":
        return height_map_column_first

    if integration_method == "average":
        # stack both to then take average
        return np.stack((height_map_row_first, height_map_column_first), axis=2).mean(axis=2)

    if integration_method == "random":
        return random_path_integration(height_map_row_first, height_map_column_first, y_cumsum, x_cumsum)
```


```python
def diff_along_axis(v):
    # diff_along_x[y,i,j] denotes v[y,j] - v[y,i]
    diff_along_x = v[:,None,:] - v[:,:,None]
    # diff_along_y[n,m,x] denotes v[m,x] - v[n,x]
    diff_along_y = v[None,:,:] - v[:,None,:]

    return diff_along_y, diff_along_x

def diff_along_axis_test():
    ### unit test for diff_along_axis
    v = np.arange(30).reshape(5,6) ** 2
    diff_along_y, diff_along_x = diff_along_axis(v)
    for y in range(v.shape[0]):
            for i in range(v.shape[1]):
                for j in range(v.shape[1]):
                    assert diff_along_x[y,i,j] == v[y,j] - v[y,i]
    for x in range(v.shape[1]):
            for n in range(v.shape[0]):
                for m in range(v.shape[0]):
                    assert diff_along_y[n,m,x] == v[m,x] - v[n,x]

diff_along_axis_test()
```


```python
def traverse_random(v, d_v, axis):
    '''
    For each point in v, map it to a new point randomly along the specified axis.
    The value at the new point will the determined as v[old] + d_v[old,new]
    d_v should be in the format of diff_along_axis
    '''
    h,w = v.shape

    # fixed endpoints in row major
    y_end = np.repeat(range(h),w)
    x_end = np.tile(range(w),h)

    if axis == 0:
        # random starting point along the column
        y_start = np.random.choice(h,h*w)
        # fancy indexing to map a random start point to fixed end point
        return (v[y_start,x_end] + d_v[y_start,y_end,x_end]).reshape(h,w)

    if axis == 1:
        x_start = np.random.choice(w,h*w)
        return (v[y_end,x_start] + d_v[y_end,x_start,x_end]).reshape(h,w)

def traverse_random_test():
    h = 20
    w = 16

    v = np.arange(h * w).reshape(h,w) ** 2
    dy, dx = diff_along_axis(v)

    assert np.all(traverse_random(v, dy, 0) == v)
    assert np.all(traverse_random(v, dx, 1) == v)

traverse_random_test()
```


```python
def random_path_integration(s_row_first, s_column_first, y_cumsum, x_cumsum, samples=100, turns=1):

    y_cumsum_diffs, _ = diff_along_axis(y_cumsum)
    _, x_cumsum_diffs = diff_along_axis(x_cumsum)

    s_init = (s_column_first, s_row_first)
    d_s = (y_cumsum_diffs, x_cumsum_diffs)

    s_cum_mean = np.zeros(s_row_first.shape)

    for i in range(samples):
        
        # grab wither the column or row first
        s = s_init[i % 2]

        # how many "turns" to do
        for turn in range(turns):
            axis = (i + turn) % 2
            # build new map from traversal from the old pixel
            s = traverse_random(s, d_s[axis], axis)
        
        s_cum_mean = (i/(i+1)) * s_cum_mean + (1/(i+1)) * s

    return s_cum_mean
```

# Main function


```python
def plot_subject(subject_name, integration_methods=('row','column','average','random'), time_execution_samples=1):
    root_path = '/content/drive/MyDrive/croppedyale/'
    full_path = '%s%s' % (root_path, subject_name)
    ambient_image, imarray, light_dirs = LoadFaceImages(full_path, subject_name, 64)

    processed_imarray = preprocess(ambient_image, imarray)

    albedo_image, surface_normals = photometric_stereo(processed_imarray,
                                                    light_dirs)
    
    plot_surface_normals(surface_normals, subject_name)
    display_albedo(albedo_image, subject_name)

    for integration_method in integration_methods:
        
        start = time.time()
        for i in range(time_execution_samples):
            height_map = get_surface(surface_normals, integration_method)
        end = time.time()

        print (f"integration_method={integration_method},avg_time_elapsed={(end-start)/time_execution_samples}")
        display_height_3d(albedo_image, height_map, f"subject_name={subject_name},integration_method={integration_method}")
```


```python
plot_subject('yaleB01', ['random'])
```

    integration_method=random,avg_time_elapsed=0.2918431758880615



    
![png](mp1_files/mp1_20_1.png)
    



    
![png](mp1_files/mp1_20_2.png)
    



    
![png](mp1_files/mp1_20_3.png)
    



```python
plot_subject('yaleB02', ['random'])
```

    integration_method=random,avg_time_elapsed=0.26366138458251953



    
![png](mp1_files/mp1_21_1.png)
    



    
![png](mp1_files/mp1_21_2.png)
    



    
![png](mp1_files/mp1_21_3.png)
    



```python
plot_subject('yaleB05', ['random'])
```

    integration_method=random,avg_time_elapsed=0.2736051082611084



    
![png](mp1_files/mp1_22_1.png)
    



    
![png](mp1_files/mp1_22_2.png)
    



    
![png](mp1_files/mp1_22_3.png)
    



```python
plot_subject('yaleB07',time_execution_samples=5)
```

    integration_method=row,avg_time_elapsed=0.000467681884765625
    integration_method=column,avg_time_elapsed=0.00043082237243652344
    integration_method=average,avg_time_elapsed=0.001269960403442383
    integration_method=random,avg_time_elapsed=0.2546403408050537



    
![png](mp1_files/mp1_23_1.png)
    



    
![png](mp1_files/mp1_23_2.png)
    



    
![png](mp1_files/mp1_23_3.png)
    



    
![png](mp1_files/mp1_23_4.png)
    



    
![png](mp1_files/mp1_23_5.png)
    



    
![png](mp1_files/mp1_23_6.png)
    


Following is using a subset of yaleB02, deleting images where the nose casts a considerable shadow.


```python
plot_subject('yaleB12', ['random'])
```

    Total available images is less than specified.
    Proceeding with 48 images.
    
    integration_method=random,avg_time_elapsed=0.2618544101715088



    
![png](mp1_files/mp1_25_1.png)
    



    
![png](mp1_files/mp1_25_2.png)
    



    
![png](mp1_files/mp1_25_3.png)
    



```python

```
