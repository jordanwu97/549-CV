# MP0 : Image Demosaicing

Welcome to CS 543! This assignment is a warm-up assignment to get you back up working from the winter break! We will try to provide you an iPython Notebook (like this) for all the future assignments! The notebook will provide you some further instructions(implementation related mainly), in addition to the ones provided on class webpage.

### Import statements

The following cell is only for import statements. You can use any of the 3 : cv2, matplotlib or skimage for image i/o and other functions. We will provide you the names of the relevant functions for each module. __{For convenience provided at the end of the class assignment webpage}__


```python
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import skimage
import scipy
from scipy import ndimage
from scipy import signal

%matplotlib inline
plt.rcParams['figure.figsize'] = 10,10
```

### Reading the Mosaic Image


```python
IMG_DIR = 'images/'
def read_image(IMG_NAME):
    # YOUR CODE HERE
    img = cv2.imread(IMG_DIR + IMG_NAME, 1)
    return img
```


```python
IMG_NAME = 'crayons.bmp'
mosaic_img = read_image(IMG_NAME)# YOUR CODE HERE
```


```python
# sanity checking image is loaded properly
plt.imshow(mosaic_img[:,:,0])
```




    <matplotlib.image.AxesImage at 0x7fc8ee405520>




    
![png](mp0_files/mp0_6_1.png)
    


### Linear Interpolation


```python
def bayer_mask(shape):
    # Return masks of shape (shape[0],shape[1],3) for R,G,B
    # ex: bayer_mask(mosaic_img.shape)[:,:,1] for G mask
    
    R_TILE = [[1,0],[0,0]]
    G_TILE = [[0,1],[1,0]]
    B_TILE = [[0,0],[0,1]]
    
    y,x = shape
    
    mask = np.zeros((y,x,3), dtype=int)
    
    for c, tile in enumerate([R_TILE, G_TILE, B_TILE]):
        mask[:,:,c] = np.tile(tile, (y//2 + 1,x//2 + 1))[:y,:x]
    
    return mask

def rgb_2_bgr(rgb):
    return rgb[:,:,[2,1,0]]

def bgr_2_rgb(bgr):
    return bgr[:,:,[2,1,0]]
```


```python
def get_solution_image(mosaic_img):
    '''
    This function should return the soln image.
    Feel free to write helper functions in the above cells
    as well as change the parameters of this function.
    '''
    
    mosaic_img = mosaic_img[:,:,0]
    
    mosaic_shape = np.shape(mosaic_img)
    soln_image = np.zeros((mosaic_shape[0], mosaic_shape[1], 3), dtype=int)
    ### YOUR CODE HERE ###
    
    # weights for correlation
    weights = np.zeros((3,3,3),dtype=int)
    weights[0] = np.ones((3,3),dtype=int) # get all 9 cells for red
    weights[1] = [[0,1,0],[1,1,1],[0,1,0]] # t shape filter for green
    weights[2] = np.ones((3,3),dtype=int) # get all 9 cells for blue
    
    
    mosaic_bayer_mask = bayer_mask(mosaic_shape)
    
    # separate r,g,b into 3 layers using mask
    mosaic_bayer_rgb = mosaic_bayer_mask * mosaic_img[:,:,None]
    
    # do for r,g,b layers
    for c in range(3):
        summed = ndimage.correlate(mosaic_bayer_rgb[:,:,c], weights[c], mode="reflect")
        # divide by number of non-zeros of mask to get mean
        divisor = ndimage.correlate(mosaic_bayer_mask[:,:,c], weights[c], mode="reflect")
        soln_image[:,:,c] = summed // divisor
    
    return rgb_2_bgr(soln_image)

soln = get_solution_image(mosaic_img)
plt.imshow(bgr_2_rgb(soln))
# convert to bgr before writing
cv2.imwrite("tmp.png", soln)
```




    True




    
![png](mp0_files/mp0_9_1.png)
    



```python
from scipy import stats

def compute_errors(soln_image, original_image):
    '''
    Compute the Average and Maximum per-pixel error
    for the image.
    
    Also generate the map of pixel differences
    to visualize where the mistakes are made
    '''
    
    se_layer = np.square(soln_image - original_image)
    
#     for i in range(0,3):
#         print (np.sqrt(np.max(se_layer[:,:,i])))
    
    se = np.sum(se_layer, axis=2)
    
    pp_err = np.mean(se)
    max_err = np.max(se)
    
    se_show = np.power(se,1/4)
    
    plt.imshow(se_show,cmap='gray')
    plt.title("error map")

    
    return pp_err, max_err


```


```python
def show_compare_patch(soln_image, original_image, yrange, xrange, shape=(1,2)):
    plt.figure()
    f, axarr = plt.subplots(shape[0],shape[1]) 
    axarr[0].imshow(bgr_2_rgb(soln_image[yrange[0]:yrange[1],xrange[0]:xrange[1]]))
    axarr[0].title.set_text("soln")
    axarr[1].imshow(bgr_2_rgb(original_image[yrange[0]:yrange[1],xrange[0]:xrange[1]]))
    axarr[1].title.set_text("original")
```

We provide you with 3 images to test if your solution works. Once it works, you should generate the solution for test image provided to you.


```python
def demosaic_and_compare(img_name, patch, demosaicing_func):
    
    mosaic_img = read_image(f'{img_name}.bmp')
    soln_image = demosaicing_func(mosaic_img)
    original_image = read_image(f'{img_name}.jpg')
    # For sanity check display your solution image here
    ### YOUR CODE
    plt.imshow(bgr_2_rgb(soln_image))
    plt.title("demosaiced solution")
    plt.figure()
    
    pp_err, max_err = compute_errors(soln_image, original_image)
    print(f"The average per-pixel error for {img_name} is: "+str(pp_err))
    print(f"The maximum per-pixel error for {img_name} is: "+str(max_err))
    plt.figure()
    
    yrange, xrange = patch
    
    show_compare_patch(soln_image,original_image, yrange, xrange)
    
    
```


```python
demosaic_and_compare("crayons", ((-200,-100),(None,200)), get_solution_image)
```

    The average per-pixel error for crayons is: 143.4874548611111
    The maximum per-pixel error for crayons is: 53633



    
![png](mp0_files/mp0_14_1.png)
    



    
![png](mp0_files/mp0_14_2.png)
    



    <Figure size 720x720 with 0 Axes>



    <Figure size 720x720 with 0 Axes>



    
![png](mp0_files/mp0_14_5.png)
    



```python
demosaic_and_compare("iceberg", ((1000,1100),(1200,1300)), get_solution_image)
```

    The average per-pixel error for iceberg is: 105.24193808890755
    The maximum per-pixel error for iceberg is: 30186



    
![png](mp0_files/mp0_15_1.png)
    



    
![png](mp0_files/mp0_15_2.png)
    



    <Figure size 720x720 with 0 Axes>



    <Figure size 720x720 with 0 Axes>



    
![png](mp0_files/mp0_15_5.png)
    



```python
demosaic_and_compare("tony", ((50,250),(500,700)), get_solution_image)
```

    The average per-pixel error for tony is: 23.654376041666666
    The maximum per-pixel error for tony is: 9640



    
![png](mp0_files/mp0_16_1.png)
    



    
![png](mp0_files/mp0_16_2.png)
    



    <Figure size 720x720 with 0 Axes>



    <Figure size 720x720 with 0 Axes>



    
![png](mp0_files/mp0_16_5.png)
    


We see interpolation start to fall apart on small details and edges. They become quite fuzzy with incorrect splotches of colors. This is expected since the interpolation masks act as a blur filter.


```python
mosaic_img = read_image('hope.bmp')
soln_image = get_solution_image(mosaic_img)
# Generate your solution image here and show it 
plt.imshow(bgr_2_rgb(soln_image))
cv2.imwrite("hope_interpolation.png", soln_image)
```




    True




    
![png](mp0_files/mp0_18_1.png)
    


### Freeman's Method

For details of the freeman's method refer to the class assignment webpage.

__MAKE SURE YOU FINISH LINEAR INTERPOLATION BEFORE STARTING THIS PART!!!__


```python
def get_freeman_solution_image(mosaic_img):
    '''
    This function should return the freeman soln image.
    Feel free to write helper functions in the above cells
    as well as change the parameters of this function.
    
    HINT : Use the above get_solution_image function.
    '''
    ### YOUR CODE HERE ###
    
    mosaic_img = get_solution_image(mosaic_img).astype(float)
    
    freeman_soln_image = np.copy(mosaic_img)
    
    # mosaic_img is bgr
    
    for c in [0,2]:
        c_g = mosaic_img[:,:,c] - mosaic_img[:,:,1]
        filt = scipy.signal.medfilt2d(c_g)
        freeman_soln_image[:,:,c] = filt + mosaic_img[:,:,1]

    return np.clip(freeman_soln_image.astype(int),a_min=0,a_max=255)
```


```python
demosaic_and_compare("crayons", ((-200,-100),(None,200)), get_freeman_solution_image)
```

    The average per-pixel error for crayons is: 100.47626736111111
    The maximum per-pixel error for crayons is: 48173



    
![png](mp0_files/mp0_21_1.png)
    



    
![png](mp0_files/mp0_21_2.png)
    



    <Figure size 720x720 with 0 Axes>



    <Figure size 720x720 with 0 Axes>



    
![png](mp0_files/mp0_21_5.png)
    



```python
demosaic_and_compare("iceberg", ((1000,1100),(1200,1300)), get_freeman_solution_image)
```

    The average per-pixel error for iceberg is: 66.46696724274652
    The maximum per-pixel error for iceberg is: 34075



    
![png](mp0_files/mp0_22_1.png)
    



    
![png](mp0_files/mp0_22_2.png)
    



    <Figure size 720x720 with 0 Axes>



    <Figure size 720x720 with 0 Axes>



    
![png](mp0_files/mp0_22_5.png)
    


The Freeman one looks much better than the interpolated one above.


```python
demosaic_and_compare("tony", ((50,250),(500,700)), get_freeman_solution_image)
```

    The average per-pixel error for tony is: 15.451165625
    The maximum per-pixel error for tony is: 10753



    
![png](mp0_files/mp0_24_1.png)
    



    
![png](mp0_files/mp0_24_2.png)
    



    <Figure size 720x720 with 0 Axes>



    <Figure size 720x720 with 0 Axes>



    
![png](mp0_files/mp0_24_5.png)
    



```python
mosaic_img = read_image('hope.bmp')
soln_image = get_freeman_solution_image(mosaic_img)
# Generate your solution image here and show it
plt.imshow(bgr_2_rgb(soln_image))
cv2.imwrite("hope_freeman.png", soln_image)
```




    True




    
![png](mp0_files/mp0_25_1.png)
    


### Mosaicing an Image

Now lets take a step backwards and mosaic an image.


```python
def get_mosaic_image(original_image):
    '''
    Generate the mosaic image using the Bayer Pattern.
    '''
    
    shape = original_image[:,:,0].shape
    
    # apply all 3 bayer mask to the original image, then flatten it to 1 layer
    mosaic = np.sum(bgr_2_rgb(original_image) * bayer_mask(shape), axis=2)
    
    # copy since mosaic_img has 3 channels
    soln = np.zeros(original_image.shape, dtype=int)
    soln[:,:,0] = mosaic
    soln[:,:,1] = mosaic
    soln[:,:,2] = mosaic
    
    return soln
```


```python
### YOU CAN USE ANY OF THE PROVIDED IMAGES TO CHECK YOUR get_mosaic_function
original_image = read_image('crayons.jpg')
mosaiced = get_mosaic_image(original_image)
mosaic_img = read_image('crayons.bmp')
print ("get_mosaic_image matches bmp:", np.all(mosaiced == mosaic_img))
```

    get_mosaic_image matches bmp: True


Use any 3 images you find interesting and generate their mosaics as well as their demosaics. Try to find images that break your demosaicing function.


```python
### YOUR CODE HERE ###
def mosaic_and_demosaic(image_name):
    original_image = read_image(image_name)
    
    mosaiced = get_mosaic_image(original_image)
    plt.imshow(mosaiced)
    plt.title("mosaiced")
    plt.figure()
    
    soln_image = get_solution_image(mosaiced)
    plt.imshow(bgr_2_rgb(soln_image))
    plt.title("soln")
    plt.figure()
    
    pp_err, max_err = compute_errors(soln_image, original_image)
    print(f"The average per-pixel error for {image_name} is: "+str(pp_err))
    print(f"The maximum per-pixel error for {image_name} is: "+str(max_err))
```


```python
mosaic_and_demosaic("street.jpg")
```

    The average per-pixel error for street.jpg is: 201.67202010391821
    The maximum per-pixel error for street.jpg is: 21517



    
![png](mp0_files/mp0_32_1.png)
    



    
![png](mp0_files/mp0_32_2.png)
    



    
![png](mp0_files/mp0_32_3.png)
    



```python
mosaic_and_demosaic("vessel.jpg")
```

    The average per-pixel error for vessel.jpg is: 684.1286760416667
    The maximum per-pixel error for vessel.jpg is: 50634



    
![png](mp0_files/mp0_33_1.png)
    



    
![png](mp0_files/mp0_33_2.png)
    



    
![png](mp0_files/mp0_33_3.png)
    



```python
mosaic_and_demosaic("nature.jpg")
```

    The average per-pixel error for nature.jpg is: 269.2834053760393
    The maximum per-pixel error for nature.jpg is: 33650



    
![png](mp0_files/mp0_34_1.png)
    



    
![png](mp0_files/mp0_34_2.png)
    



    
![png](mp0_files/mp0_34_3.png)
    


### Bonus Points


```python
### YOUR CODE HERE ###
### YOU ARE ON YOUR OWN :) ####
```
