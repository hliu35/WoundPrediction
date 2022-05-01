import os, sys
import shutil

import cv2
from matplotlib.pyplot import show
import numpy as np

outpath = "data_cropped/"

def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname,img)


def create_circular_mask(h, w, center=None, radius=None):
    # https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def mask_and_enlarge(img, mask, shape, center, r):
    img_1 = img.copy()
    img_1[~mask, :] = 0

    a, b = center
    img_1 = img_1[b-r:b+r, a-r:a+r, :]
    img_1 = cv2.resize(img_1, shape)
    return img_1


def crop_with_cv2(filepath):

    file = filepath.split("/")[-1]
    path = os.path.join(outpath, file)

    # Read image.
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    detected_circles = cv2.HoughCircles(gray, 
                    cv2.HOUGH_GRADIENT, 1, 50, param1 = 60,
                param2 = 40, minRadius = 40, maxRadius = 100)

    r = 80 # assign a random number

    # Draw circles that are detected.
    if detected_circles is not None:
    
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
    
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            if a < r or b < r: continue # the circle cannot go out of bound

            mask = create_circular_mask(h=img.shape[0], w=img.shape[1], center=(a, b), radius=r)
            img_1 = mask_and_enlarge(img, mask, img.shape[:2], center=(a, b), r=r)

            img_2 = img.copy()
            # Draw the circumference of the circle.
            cv2.circle(img_2, (a, b), r, (0, 255, 0), 2)
    
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img_2, (a, b), 1, (0, 0, 255), 3)
            showInMovedWindow("Detected Circle", img_2, 1000, 400)

            k = cv2.waitKey(0)
            if k == 13:    # S key to stop
                cv2.imwrite(path, img_1)
                cv2.destroyAllWindows()
                return
            
            else:
                continue # else print its value
    
    cv2.destroyAllWindows()

    r = 90
    mask = create_circular_mask(h=img.shape[0], w=img.shape[1], radius=r)
    img_1 = mask_and_enlarge(img, mask, img.shape[:2], center=(img.shape[0]//2, img.shape[1]//2), r=r)

    showInMovedWindow("no circle detected or chosen", img_1, 1000, 400)  
    cv2.waitKey(0)
    cv2.imwrite(path, img_1)
    cv2.destroyAllWindows()



if __name__ == "__main__":

    inpath = "data/"

    # create output folder
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.mkdir(outpath)

    filelist = [f for f in os.listdir("data/") if ".png" in f]
    print(filelist[:10])
    
    for f in filelist:
        fp = os.path.join(inpath, f)
        crop_with_cv2(fp)