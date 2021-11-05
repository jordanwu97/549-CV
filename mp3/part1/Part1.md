# Part 1: Stitching pairs of images

### Setup


```python
!wget -qc http://slazebni.cs.illinois.edu/fall21/assignment3/left.jpg http://slazebni.cs.illinois.edu/fall21/assignment3/right.jpg
```


```python
!pip install -q --upgrade opencv-contrib-python==4.5.4.58

# imports
import numpy as np
import skimage
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance
import scipy
import skimage.transform
from more_itertools import grouper
from itertools import combinations

cv2.__version__
```

    [K     |████████████████████████████████| 66.5 MB 7.0 kB/s 
    [?25h




    '4.5.4-dev'




```python
def load_im(filename):
    return cv2.imread(filename,1)[:,:,::-1] / 255

def pad(im,top=0,left=0,bottom=0,right=0):
    if len(im.shape) == 3:
        h,w,d = im.shape
        canvas = np.zeros((h+top+bottom,w+left+right,d))
        canvas[top:top+h,left:left+w,:] = im
        return canvas
    else:
        h,w = im.shape
        canvas = np.zeros((h+top+bottom,w+left+right))
        canvas[top:top+h,left:left+w] = im
        return canvas
```


```python
# Provided code - nothing to change here

def plot_inlier_matches(ax, img1, img2, inliers):
    """
    Plot the matches between two images according to the matched keypoints
    :param ax: plot handle
    :param img1: left image
    :param img2: right image
    :inliers: x,y in the first image and x,y in the second image (Nx4)
    """
    res = np.hstack([img1, img2])
    ax.set_aspect('equal')
    ax.imshow(res, cmap='gray')
    
    ax.plot(inliers[:,0], inliers[:,1], '+r')
    ax.plot(inliers[:,2] + img1.shape[1], inliers[:,3], '+r')
    ax.plot([inliers[:,0], inliers[:,2] + img1.shape[1]],
            [inliers[:,1], inliers[:,3]], 'r', linewidth=0.4)
    ax.axis('off')
    
# Usage:
# fig, ax = plt.subplots(figsize=(20,10))
# plot_inlier_matches(ax, img1, img2, computed_inliers)
```

### Feature Detection


```python
# See assignment page for the instructions!
def sift_detect(img,ax=None,sift=cv2.SIFT_create(1000)):
    kps, des = sift.detectAndCompute(img, None)
    if ax != None:
        im_draw = cv2.drawKeypoints(img, kps, np.empty_like(img), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax.imshow(im_draw)
        ax.axis("off")
    kps_xy = np.array(list(np.array(kp.pt) for kp in kps))
    return kps_xy, des
```


```python
def putative_pairs(left_desc, right_desc, reject_ambiguous_thresh):
    desc_dist = scipy.spatial.distance.cdist(left_desc, right_desc, 'sqeuclidean')

    # find where difference in dist between top 2 picks is large to reject ambiguous matches
    top_2_diff = np.abs(np.sum(np.partition(desc_dist, 2, axis=1)[:,:2] * np.array([[1,-1]]), axis=1))
    left_select = np.argwhere(top_2_diff > reject_ambiguous_thresh)[:,0]
    right_select = np.argmin(desc_dist[left_select],axis=1)

    selections_ambiguous = np.c_[left_select,right_select]

    # sort selections by descriptor distance
    sorted_by_min = selections_ambiguous[np.argsort(desc_dist[left_select, right_select])]

    return sorted_by_min
```


```python
def to_homogeneous(pts):
    return np.hstack((pts,np.ones(len(pts))[:,None]))

def from_homogeneous(homo):
    return homo[:,:2] / homo[:,[2]]

def matA(left_right_pair):
    n = len(left_right_pair)
    left = left_right_pair[:,:2]
    right = left_right_pair[:,2:]

    left_homo = to_homogeneous(left)

    A = np.zeros((n,2,9))
    A[:,0,[0,1,2,6,7,8]] = left_homo[:,[0,1,2,0,1,2]]
    A[:,0,[6,7,8]] = A[:,0,[6,7,8]] * right[:,[0]] * -1
    A[:,1,[3,4,5,6,7,8]] = left_homo[:,[0,1,2,0,1,2]]
    A[:,1,[6,7,8]] = A[:,1,[6,7,8]] * right[:,[1]] * -1

    return A
```


```python
class BatchHomography:
    def __init__(self, left_right_pair):
        self.N = len(left_right_pair)
        self.left = left_right_pair[:,:2]
        self.right = left_right_pair[:,2:]

        # initialize cached A matrix
        self.A_cached = matA(left_right_pair)

    def call(self, x):
        # returns MxNx2
        # M is number of H matrix created len(selections) in fit_many
        # N is len(x)

        return np.transpose(
            from_homogeneous(
                np.tensordot(
                    to_homogeneous(x), self.H_many.T, axes=((1),(0))
                )
            ), (2,0,1))

    def fit(self, selections):
        # Simultaneously fit many N selections of 4 points
        # selections have shape nx4
        n, num_pts = selections.shape
        A = self.A_cached[selections,:,:].reshape(n,num_pts*2,9)
        _,_,V = np.linalg.svd(A)
        self.H_many = (V[:,-1,:] / V[:,-1,[-1]]).reshape(n,3,3)

    def residuals(self):
        predict = self.call(self.left)
        return np.linalg.norm(predict - self.right, axis=2) ** 2
```


```python
def test_homography(args=np.random.rand(3)):

    angle = args[0] * 2 * np.pi

    H_true = np.array([[np.cos(angle), -1 * np.sin(angle), args[1]],
                       [np.sin(angle),  np.cos(angle), args[2]],
                       [0         , 0          , 1]])
    
    left = np.array([[1,1],
                     [1,5],
                     [4,1],
                     [4,5]])
            
    right = from_homogeneous(to_homogeneous(left) @ H_true.T)

    left_right_pair = np.hstack((left,right))

    h = BatchHomography(left_right_pair)

    selections = np.random.random(size=(10,4)).argsort(axis=1)

    h.fit(selections)

    assert np.allclose(h.residuals(), 0)

test_homography()
```


```python
def batch_ransac(model: BatchHomography, num_iter, inlier_residual_thresh, batch_size):
    
    def ransac_iter():
        for iter in range(num_iter):
            if iter % 100 == 0:
                print ("iter:",iter)

            # select batch_size x 4 points for fitting
            pt_selection = np.random.random(size=(batch_size, model.N)).argpartition(4,axis=1)[:,:4]
            pt_selection = np.unique(pt_selection, axis=0)        

            model.fit(pt_selection)
            # find expected transformation y_predict = H@x
            residuals = model.residuals()
            # residuals shape = batch_size x num_points

            # find model from batch with max number of inliers
            best_model_idx = np.argmax(np.sum(residuals < inlier_residual_thresh, axis=1))

            inliers = np.argwhere(residuals[best_model_idx] < inlier_residual_thresh)[:,0]

            yield (inliers, np.mean(residuals[best_model_idx][inliers]), model.H_many[best_model_idx])
            
    inliers,residuals,H = max(ransac_iter(), key=lambda result: (len(result[0]), -result[1]))
    # I realize the real ransac need to refit with all inliers, 
    # but I found just using the 4 point model gave better results
    # I gave an explaination in the report
    return inliers,residuals,H
    # refit using all inliers
    model.fit(inliers[None,:])
    return inliers,model.residuals()[0][inliers],model.H_many[0]
```


```python
def draw_canvas(canvas,im,mask=None):
    h,w = im.shape[:2]
    yy,xx = np.mgrid[:h,:w]
    if mask is not None:
        yy,xx = np.argwhere(mask > 0).T
    canvas[yy,xx] = im[yy,xx]
```


```python
def stitch_panorama(left_color, right_color, left_mask, canvas_shape, cache=None, inlier_residual_thresh=20):
    left_gray = cv2.cvtColor((left_color * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor((right_color * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    left_kp, left_desc = sift_detect(left_gray)
    right_kp, right_desc = sift_detect(right_gray)

    putative = putative_pairs(left_desc, right_desc, 30_000)[:300]
    putative_left_to_right_pts = np.concatenate((left_kp[putative[:,0]], right_kp[putative[:,1]]), axis=1)
    print (f"# Putative: {len(putative_left_to_right_pts)}")
    
    if cache is not None:
        inliers,residuals,H = cache[:3]
    else:
        inliers,residuals,H = batch_ransac(BatchHomography(putative_left_to_right_pts), num_iter=200, inlier_residual_thresh=inlier_residual_thresh, batch_size=500)

    print (f"# Inliers: {len(inliers)}")
    print (f"Avg Residuals (Squared Distance): {residuals:.06f}")

    canvas = np.zeros(canvas_shape + (3,))
    canvas_mask = np.zeros(canvas.shape[:2])

    right_warp = skimage.transform.warp(right_color, skimage.transform.ProjectiveTransform(H), output_shape=canvas_shape)
    right_mask = skimage.transform.warp(np.ones(right_color.shape[:2]), skimage.transform.ProjectiveTransform(H), output_shape=canvas_shape)
    
    # make the mask smaller since the edges get a bit wonky
    left_mask = (scipy.ndimage.gaussian_filter(left_mask, 1.5, mode='constant', cval=0) > 0.95) * 1

    draw_canvas(canvas, right_warp)
    draw_canvas(canvas, left_color, mask=left_mask)

    draw_canvas(canvas_mask, right_mask)
    draw_canvas(canvas_mask, left_mask, mask=left_mask)

    return inliers,residuals,H,(canvas,canvas_mask),putative_left_to_right_pts
```


```python
left = load_im("left.jpg")
left_mask = np.ones(left.shape[:2])
right = load_im("right.jpg")
C = stitch_panorama(left,
                    right,
                    left_mask,
                    (400,1300),
                    cache=None,
                    inlier_residual_thresh=15**2)
# show inliers match
fig, ax = plt.subplots(figsize=(20,10))
plot_inlier_matches(ax, left, right, C[4][C[0]])
plt.show()
# show canvas
fig, ax = plt.subplots(figsize=(20,10))
ax.imshow(C[3][0])
```

    # Putative: 129
    iter: 0
    iter: 100
    # Inliers: 65
    Avg Residuals (Squared Distance): 67.191716



    
![png](Part1_files/Part1_15_1.png)
    





    <matplotlib.image.AxesImage at 0x7fbbe5e2c750>




    
![png](Part1_files/Part1_15_3.png)
    


## Extra Credit


```python
left = load_im("/content/drive/MyDrive/549/mp3/MP3_part1_data/hill/1.JPG")
mid = load_im("/content/drive/MyDrive/549/mp3/MP3_part1_data/hill/2.JPG")
left_mask = np.ones(left.shape[:2])
right = load_im("/content/drive/MyDrive/549/mp3/MP3_part1_data/hill/3.JPG")
C12 = stitch_panorama(pad(left,top=50),
                      mid,
                      pad(left_mask,top=50),
                      (350,800),
                      cache=None,
                      inlier_residual_thresh=2)
C123 = stitch_panorama(C12[3][0],
                       right,
                       C12[3][1],
                       (350,800),
                       cache=None,
                       inlier_residual_thresh=2)
fig, ax = plt.subplots(figsize=(20,10))
ax.imshow(C123[3][0])
```

    # Putative: 300
    iter: 0
    iter: 100
    # Inliers: 300
    Avg Residuals (Squared Distance): 0.040720
    # Putative: 300
    iter: 0
    iter: 100
    # Inliers: 300
    Avg Residuals (Squared Distance): 0.064080





    <matplotlib.image.AxesImage at 0x7fbbe5eff690>




    
![png](Part1_files/Part1_17_2.png)
    



```python
left = load_im("/content/drive/MyDrive/549/mp3/MP3_part1_data/ledge/1.JPG")
mid = load_im("/content/drive/MyDrive/549/mp3/MP3_part1_data/ledge/2.JPG")
left_mask = np.ones(left.shape[:2])
right = load_im("/content/drive/MyDrive/549/mp3/MP3_part1_data/ledge/3.JPG")
Ledge_C12 = stitch_panorama(pad(left,top=400),
                      mid,
                      pad(left_mask,top=400),
                      (900,1100),
                      cache=None,
                      inlier_residual_thresh=2)
Ledge_C123 = stitch_panorama(Ledge_C12[3][0],
                       right,
                       Ledge_C12[3][1],
                       (900,1100),
                       cache=None,
                       inlier_residual_thresh=2)
fig, ax = plt.subplots(figsize=(20,10))
ax.imshow(Ledge_C123[3][0])
```

    # Putative: 277
    iter: 0
    iter: 100
    # Inliers: 207
    Avg Residuals (Squared Distance): 0.630591
    # Putative: 126
    iter: 0
    iter: 100
    # Inliers: 74
    Avg Residuals (Squared Distance): 0.592246





    <matplotlib.image.AxesImage at 0x7fbbe5fd4b10>




    
![png](Part1_files/Part1_18_2.png)
    



```python
left = load_im("/content/drive/MyDrive/549/mp3/MP3_part1_data/pier/1.JPG")
mid = load_im("/content/drive/MyDrive/549/mp3/MP3_part1_data/pier/2.JPG")
left_mask = np.ones(left.shape[:2])
right = load_im("/content/drive/MyDrive/549/mp3/MP3_part1_data/pier/3.JPG")
pier_C12 = stitch_panorama(left,
                      mid,
                      left_mask,
                      (450,1280),
                      cache=None,
                      inlier_residual_thresh=1)
pier_C123 = stitch_panorama(pier_C12[3][0],
                       right,
                       pier_C12[3][1],
                       (450,1280),
                       cache=None,
                       inlier_residual_thresh=1)
fig, ax = plt.subplots(figsize=(20,10))
ax.imshow(pier_C123[3][0])
```

    # Putative: 257
    iter: 0


    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:5: RuntimeWarning: divide by zero encountered in true_divide
      """


    iter: 100
    # Inliers: 181
    Avg Residuals (Squared Distance): 0.132477
    # Putative: 154
    iter: 0
    iter: 100
    # Inliers: 92
    Avg Residuals (Squared Distance): 0.084492





    <matplotlib.image.AxesImage at 0x7fbbe628b510>




    
![png](Part1_files/Part1_19_4.png)
    



```python

```
