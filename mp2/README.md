# Assignment 2: Hybrid Images and Scale-space blob detection


```python
# Libraries you will find useful
import numpy as np
import scipy 
import skimage
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import cv2
import scipy.ndimage
import skimage.transform
import skimage.filters
from matplotlib.patches import Circle
from scipy.ndimage.filters import gaussian_laplace
import time
```

## Part 1: Hybrid Images


```python
def load_img(fname, mode=0):
    return cv2.imread(f"/content/drive/MyDrive/549/mp2/{fname}", mode) / 255
```


```python
# matplotlib helpers
def draw_img(ax, img):
    ax.axis("off")
    if (len(img.shape) == 3):
        ax.imshow(img[:,:,::-1])
    else:
        ax.imshow(img, cmap="gray")

def img_grid_resize(fig, img, scale=1.5, grid=(1,1)):
    aspect = figure.figaspect(img)
    fig.set_size_inches(aspect * np.array(grid[::-1]) * scale)

def display_bw(*imgs, scale=1.5, labels=None, grid=None):
    h,w = imgs[0].shape[:2]
    n = len(imgs)
    grid = (1,n) if grid == None else grid
    fig, axs = plt.subplots(grid[0],grid[1])
    img_grid_resize(fig, imgs[0], scale, grid=grid)
    for img, ax in zip(imgs, np.array(axs).flatten()):
        draw_img(ax, img)

def pad_axes_img(ax,pad):
    a,b = ax.get_ylim()
    ax.set_ylim((a+pad,b-pad))
    a,b = ax.get_xlim()
    ax.set_xlim((a-pad,b+pad))
```


```python
# To display the detected regions as circle
def show_all_circles(image, cx, cy, rad, color='r', label=""):
    """
    image: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs
    """
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect('equal')
    ax.imshow(image, cmap='gray')
    ax.axis("off")
    plt.title(f'{len(cx)} circles ({label})')
    for x, y, r in zip(cx, cy, rad):
        circ = Circle((x, y), r, color=color, fill=False)
        ax.add_patch(circ)
```


```python
def gaussian_filter(img, sigma):
    # gaussian filter that handles all channels
    with_chan = np.atleast_3d(img)
    out = np.ndarray(with_chan.shape)

    for i in range(with_chan.shape[2]):
        out[:,:,i] = scipy.ndimage.gaussian_filter(with_chan[:,:,i], sigma)
    
    return out if len(img.shape) == 3 else out[:,:,0]

def hpf(img, sigma):
    return img - gaussian_filter(img, sigma)

def generate_hybrid(img_low, img_high, sigma_low, sigma_high, high_multiplier):

    # Use your intuition and trial and error to determine good values of σ for the high-pass and low-pass filters
    # One of the σ's should always be higher than the other (which one?), but the optimal values can vary from image to image.

    # sigma_low must be larger so that the cut-off frequency is lower
    assert sigma_low > sigma_high

    # Apply a low-pass filter, i.e., a standard 2D Gaussian filter, on the first (smooth) image. 
    filtered_low = gaussian_filter(img_low, sigma_low)

    # Apply a high-pass filter on the second image. 
    # The paper suggests using an impulse (identity) filter minus a Gaussian filter for this operation.
    filtered_high = hpf(img_high, sigma_high)
    # filtered_high has a range of [-1,1]

    display_bw(filtered_low, np.clip(0.5 + filtered_high, 0, 1))

    # Add or average the tranformed images to create the hybrid image.
    # filtered_high has a range of [-1,1]. 
    # I wanted to accentuate the high pass image so I gave it a multiplier to have more impact
    return np.clip(filtered_low + filtered_high * high_multiplier, 0, 1)

def display_hybrid(fname_low, fname_high, sigma_low, sigma_high, high_multiplier=1.5, color=0):
    low = load_img(fname_low, color)
    high = load_img(fname_high, color)
    display_bw(low, high)
    # Add or average the tranformed images to create the hybrid image.
    hybrid = generate_hybrid(low, high, sigma_low, sigma_high, high_multiplier)

    # draw big and small hybrid images
    fig, (big, small) = plt.subplots(1,2)
    draw_img(big, hybrid)
    draw_img(small, hybrid)
    pad_axes_img(small, max(low.shape))

    img_grid_resize(fig, hybrid, grid=(1,2))

    print ("LPF Sigma:", sigma_low)
    print ("HPF Sigma:", sigma_high)

    return hybrid
```


```python
hybrid_cereal = display_hybrid("c2.jpg","c1.jpg", 5, 1, high_multiplier=2.5)
```

    LPF Sigma: 5
    HPF Sigma: 1



    
![png](mp2_files/mp2_7_1.png)
    



    
![png](mp2_files/mp2_7_2.png)
    



    
![png](mp2_files/mp2_7_3.png)
    



```python
horse_zebra = display_hybrid("/a1/horse.jpg","/a1/zebra.jpg", 10, 2, high_multiplier=3)
```

    LPF Sigma: 10
    HPF Sigma: 2



    
![png](mp2_files/mp2_8_1.png)
    



    
![png](mp2_files/mp2_8_2.png)
    



    
![png](mp2_files/mp2_8_3.png)
    



```python
dog_cat = display_hybrid("/a1/dog.jpg", "/a1/cat.jpg", 30, 5, high_multiplier=1)
```

    LPF Sigma: 30
    HPF Sigma: 5



    
![png](mp2_files/mp2_9_1.png)
    



    
![png](mp2_files/mp2_9_2.png)
    



    
![png](mp2_files/mp2_9_3.png)
    


## Part 2: Scale-space blob detection


```python
def test_circle(r):
    # create a test circle
    assert r < 90
    xx, yy = np.mgrid[:200, :200]
    circle = (xx - 100) ** 2 + (yy - 100) ** 2
    return (circle < r**2) * 1.0
```


```python
# Creating the Laplacian filter
# Pay careful attention to setting the right filter mask size. Hint: Should the filter width be odd or even?
```


```python
# filtering the image (two implmementations)
# one that increases filter size, and one that downsamples the image
# For timing, use time.time()
def log_filtering_increase_size(img, sigma_base, scale_factor, num_scales):

    assert scale_factor > 1

    h,w = img.shape

    scale_pyramid = np.ndarray((h,w,num_scales))

    for i in range(num_scales):
        sigma = sigma_base * (scale_factor ** i)

        # normalize LoG and square
        scale_pyramid[:,:,i] = ((sigma ** 2) * gaussian_laplace(img, sigma)) ** 2

    return scale_pyramid
```


```python
def scale_shape(shape, scale_ratio):
    assert len(shape) == 2
    return tuple(np.around(scale_ratio * np.array(shape)).astype(int))
```


```python
# cv2 is much faster than skimage
def scale_cv2(img, shape):
    return cv2.resize(img, shape[::-1], 
                      interpolation = cv2.INTER_AREA if img.shape[0] > shape[0] else cv2.INTER_CUBIC)

# baseline since this was suggested in the assignment doc
def scale_skimage(img, shape):
    return skimage.transform.resize(img, shape, anti_aliasing=True)

def log_filtering_downsample_skimage(img, sigma_base, scale_factor, num_scales):
    return log_filtering_downsample(img, sigma_base, scale_factor, num_scales, resize_func=scale_skimage)

def log_filtering_downsample(img, sigma_base, scale_factor, num_scales, resize_func=scale_cv2):

    assert scale_factor > 1

    h,w = img.shape

    scale_pyramid = np.ndarray((h,w,num_scales))

    for i in range(num_scales):

        # scale_down = lambda v: int(v // (scale_factor ** i))
        # downsample the image so that the blobs maintains scale / dimension ratio
        # h_new, w_new = scale_down(h), scale_down(w)
        
        downsized = resize_func(img, scale_shape(img.shape, 1/(scale_factor ** i)))

        # filter downsized image using sigma filter, normalize then square
        filtered = ((sigma_base ** 2) * gaussian_laplace(downsized, sigma_base)) ** 2
        
        # upsample to keep so scale pyramid stores correct size image
        
        scale_pyramid[:,:,i] = resize_func(filtered, img.shape)

        # display_bw([filtered, scale_pyramid[:,:,i] ], 10)

    
    return scale_pyramid
```


```python
# test that increasing filter size and downsampling gives similar results
def filtering_similar_test():

    circle = test_circle(20)

    sigma_base, scale_factor, scales = 5.1, 1.2, 10

    inc_size = log_filtering_increase_size(circle, sigma_base, scale_factor, scales)
    downsamp = log_filtering_downsample(circle, sigma_base, scale_factor, scales)

    def compare(i):
        sqd = (inc_size[:,:,i] - downsamp[:,:,i]) ** 2
        # output the average difference of non-zeros
        print (np.sum(sqd) / np.sum(sqd > 0))
        return sqd

    display_bw(*[ compare(i) for i in range(scales) ])

filtering_similar_test()
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:14: RuntimeWarning: invalid value encountered in double_scalars
      


    nan
    1.6396218469958892e-07
    1.4366385872743983e-07
    2.057224958930836e-07
    2.8044951933326247e-07
    3.7202488604666326e-07
    4.4312113883560336e-07
    2.7436306544433287e-07
    5.11898425995042e-07
    1.3230928506277337e-07



    
![png](mp2_files/mp2_16_2.png)
    



```python
# nonmaximum suppression in scale space
# you may find functions scipy.ndimage.filters.rank_filter or scipy.ndimage.filters.generic_filter useful

def nms_3d(scale_pyramid, thresh):
    # Perform non-maximum suppression in scale space using 3x3x3 window. 

    # Check if each cell is equal to the highest value in its 3x3x3 neighborhood.
    # If not, drop it to 0
    max_neighbors = scipy.ndimage.filters.maximum_filter(scale_pyramid, size=3)
    non_maximum_suppressed = scale_pyramid * (scale_pyramid == max_neighbors)

    return non_maximum_suppressed

def nms_2d(scale_pyramid, thresh):
    # Perform non-maximum suppression in scale space using 3x3 window first, 
    # then taking only pixels from layers that is the max across all layers  

    # find the window max in each slice 
    max_neighbors_2d = np.empty(scale_pyramid.shape)
    for i in range(scale_pyramid.shape[2]):
        max_neighbors_2d[:,:,i] = scipy.ndimage.filters.maximum_filter(scale_pyramid[:,:,i], size=3)

    # squash so get the max of all layers at each pixel
    all_layers_max = np.max(max_neighbors_2d,axis=2)

    # keep only those that matches all layer max
    non_maximum_suppressed = scale_pyramid * (scale_pyramid == all_layers_max[:,:,None])
    
    return non_maximum_suppressed

```


```python
def time_exec(f):
    start = time.time()
    res = f()
    end = time.time()
    return res, end - start

def blob_detect(img, filter_func, sigma_base, scale_factor, num_scales, thresh):

    scale_pyramid, duration = time_exec(lambda: filter_func(img, sigma_base, scale_factor, num_scales))
    print (f"{filter_func.__name__}_duration={duration:.06f}")

    # Perform non-maximum suppression. 
    non_maximum_suppressed, duration = time_exec(lambda: nms_3d(scale_pyramid, thresh))
    # print (f"nms_duration={duration:.06f}")

    # Helper for finding threshholds
    # print ("Per-layer max response", np.max(non_maximum_suppressed, axis=(0,1)))
    
    # Threshhold so values less than thresh are dropped
    threshed_nmsed_pyramid = non_maximum_suppressed * (non_maximum_suppressed > thresh)
    
    # Retreieve index of non-zero response
    picks = np.argwhere(threshed_nmsed_pyramid > 0)
    picks_x = picks[:,1]
    picks_y = picks[:,0] 
    picks_layer = picks[:,2]
    # get scale and calculate radius for responsive pixels
    picks_r = 1.414213 * (sigma_base * scale_factor ** (picks_layer))

    # print (f"num circles={len(picks_r)}")

    show_all_circles(img,picks_x,picks_y,picks_r,label=f"filter_func={filter_func.__name__}")
```


```python
def use_both_filters(accept_filter):
    for filter in [log_filtering_increase_size, log_filtering_downsample]:
        accept_filter(filter)
```


```python
# test with circles
use_both_filters(lambda f: blob_detect(test_circle(20), f, 10, 1.2, 10, 0.5))
```

    log_filtering_increase_size_duration=0.267945
    log_filtering_downsample_duration=0.039625



    
![png](mp2_files/mp2_20_1.png)
    



    
![png](mp2_files/mp2_20_2.png)
    



```python
print ("Example 1")
display_bw(load_img("assignment2_images/butterfly.jpg",1))
use_both_filters(lambda f: blob_detect(load_img("assignment2_images/butterfly.jpg"), f, 2, 1.3, 10, 0.03))
```

    Example 1
    log_filtering_increase_size_duration=0.361062
    log_filtering_downsample_duration=0.065552



    
![png](mp2_files/mp2_21_1.png)
    



    
![png](mp2_files/mp2_21_2.png)
    



    
![png](mp2_files/mp2_21_3.png)
    



```python
for idx, img_name in enumerate(["/assignment2_images/einstein.jpg", "/assignment2_images/fishes.jpg", "/assignment2_images/sunflowers.jpg"]):
    print (f"Example {idx+2}")
    display_bw(load_img(img_name,1))
    use_both_filters(lambda f: blob_detect(load_img(img_name), f, 2, 1.3, 15, 0.015))
    plt.show()
```

    Example 2
    log_filtering_increase_size_duration=2.533415
    log_filtering_downsample_duration=0.145797



    
![png](mp2_files/mp2_22_1.png)
    



    
![png](mp2_files/mp2_22_2.png)
    



    
![png](mp2_files/mp2_22_3.png)
    


    Example 3
    log_filtering_increase_size_duration=1.364185
    log_filtering_downsample_duration=0.073085



    
![png](mp2_files/mp2_22_5.png)
    



    
![png](mp2_files/mp2_22_6.png)
    



    
![png](mp2_files/mp2_22_7.png)
    


    Example 4
    log_filtering_increase_size_duration=0.940985
    log_filtering_downsample_duration=0.060123



    
![png](mp2_files/mp2_22_9.png)
    



    
![png](mp2_files/mp2_22_10.png)
    



    
![png](mp2_files/mp2_22_11.png)
    



```python
for idx, img_name in enumerate(["/assignment2_images/mural.png", "/assignment2_images/parking.png", "/assignment2_images/airport.png", "/assignment2_images/dogsled.jpg"]):
    print (f"Example {idx+5}")
    display_bw(load_img(img_name, 1), scale=1.5)
    bw = load_img(img_name)
    use_both_filters(lambda f: blob_detect(bw, f, 2, 1.3, 15, 0.03))
    plt.show()
```

    Example 5
    log_filtering_increase_size_duration=5.882798
    log_filtering_downsample_duration=0.316504



    
![png](mp2_files/mp2_23_1.png)
    



    
![png](mp2_files/mp2_23_2.png)
    



    
![png](mp2_files/mp2_23_3.png)
    


    Example 6
    log_filtering_increase_size_duration=5.763484
    log_filtering_downsample_duration=0.310438



    
![png](mp2_files/mp2_23_5.png)
    



    
![png](mp2_files/mp2_23_6.png)
    



    
![png](mp2_files/mp2_23_7.png)
    


    Example 7
    log_filtering_increase_size_duration=5.820110
    log_filtering_downsample_duration=0.314487



    
![png](mp2_files/mp2_23_9.png)
    



    
![png](mp2_files/mp2_23_10.png)
    



    
![png](mp2_files/mp2_23_11.png)
    


    Example 8
    log_filtering_increase_size_duration=6.516834
    log_filtering_downsample_duration=0.363815



    
![png](mp2_files/mp2_23_13.png)
    



    
![png](mp2_files/mp2_23_14.png)
    



    
![png](mp2_files/mp2_23_15.png)
    


Various Techniques Speed Comparisons


```python
# rand = np.random.rand(1000,600, 10)
# %timeit log_filtering_downsample(rand[:,:,0], 2, 1.3, 15)
# %timeit log_filtering_downsample_skimage(rand[:,:,0], 2, 1.3, 15)
# %timeit log_filtering_increase_size(rand[:,:,0], 2, 1.3, 15)
# %timeit scipy.ndimage.generic_filter(rand,np.max,3)
# %timeit scipy.ndimage.maximum_filter(rand,3)
# %timeit scipy.ndimage.rank_filter(rand,-1,3)
# %timeit nms_3d(rand, 0)
# %timeit nms_2d(rand, 0)
```

```
1 loop, best of 5: 305 ms per loop
1 loop, best of 5: 1.61 s per loop
1 loop, best of 5: 4.9 s per loop
1 loop, best of 5: 30.9 s per loop
1 loop, best of 5: 239 ms per loop
1 loop, best of 5: 238 ms per loop
1 loop, best of 5: 261 ms per loop
1 loop, best of 5: 342 ms per loop
```

# Extra Credit

### Colored Hybrid Images


```python
_=display_hybrid("c2.jpg","c1.jpg", 5, 1.5, color=1, high_multiplier=2.5)
plt.show()
_=display_hybrid("/a1/horse.jpg","/a1/zebra.jpg", 10, 2, high_multiplier=3, color=1)
plt.show()
_=display_hybrid("/a1/dog.jpg", "/a1/cat.jpg", 30, 3, high_multiplier=2.5, color=1)
```

    LPF Sigma: 5
    HPF Sigma: 1.5



    
![png](mp2_files/mp2_29_1.png)
    



    
![png](mp2_files/mp2_29_2.png)
    



    
![png](mp2_files/mp2_29_3.png)
    


    LPF Sigma: 10
    HPF Sigma: 2



    
![png](mp2_files/mp2_29_5.png)
    



    
![png](mp2_files/mp2_29_6.png)
    



    
![png](mp2_files/mp2_29_7.png)
    


    LPF Sigma: 30
    HPF Sigma: 3



    
![png](mp2_files/mp2_29_9.png)
    



    
![png](mp2_files/mp2_29_10.png)
    



    
![png](mp2_files/mp2_29_11.png)
    


### Seperating Low and High Image using Laplacian Pyramid


```python
ims = ( gaussian_laplace(hybrid_cereal, sigma=1.35 ** s) for s in range(10) )
display_bw(*ims, grid=(2,5))

ims = ( gaussian_laplace(horse_zebra, sigma=2 * 1.4 ** s) for s in range(10) )
display_bw(*ims, grid=(2,5))

ims = ( gaussian_laplace(dog_cat, sigma=5 * 1.2 ** s) for s in range(10) )
display_bw(*ims, grid=(2,5))
```


    
![png](mp2_files/mp2_31_0.png)
    



    
![png](mp2_files/mp2_31_1.png)
    



    
![png](mp2_files/mp2_31_2.png)
    



```python

```
