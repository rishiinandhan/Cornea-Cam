
"""
@author: rishiinandhan
"""

# Header files
from __future__ import print_function

import SimpleITK as sitk
import sys
import numpy as np
import os
import cv2 as cv
from math import atan2, cos, sin, sqrt, pi

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

def getOrientation(pts, img):
    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    
#    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
#    drawAxis(img, cntr, p1, (0, 255, 0), 1)
#    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
    print(eigenvectors)
    return angle
# Check minimum number of arguments in command line

if len( sys.argv ) < 3:
  print("Usage: Segmentation InputImage OutputImage");
  sys.exit( 1 )

# Read the image
  
reader = sitk.ImageFileReader()
reader.SetFileName( sys.argv[1] )
image = reader.Execute();

# HSV data
img_for_hsv = cv.imread(sys.argv[1]);
hsv_image = cv.cvtColor(img_for_hsv,cv.COLOR_BGR2HSV);
mask2=hsv_image[:,:,0]

# green range
lower_green = np.array([50,50,30]);
upper_green = np.array([70,255,255]);

#hue channel
lower_thr = np.array([0,0,0]);
upper_thr = np.array([179,0,0]);


mask = cv.inRange(hsv_image, lower_green, upper_green)
#mask2 = cv.inRange(hsv_image, lower_thr, upper_thr)


data = sitk.GetArrayFromImage(image);
print(data[:,:,1].shape)
image = sitk.GetImageFromArray(data[:,:,1])

writer = sitk.ImageFileWriter()
writer.SetFileName( sys.argv[3] )
writer.Execute( image )

#help(image)
#print(type(image))

# Image properties
#print(image.GetDepth())
print("Height of image : ",image.GetHeight())
#print(image.GetSize())
print("Width of image : ",image.GetWidth())
w = image.GetWidth()
h = image.GetHeight() 

# Converting RGB to binary image
# image = sitk.VectorIndexSelectionCastImageFilter(image,1)

# Do some preprocessing 

# Otsu Thresholding

otsu_filter = sitk.OtsuThresholdImageFilter()
otsu_filter.SetInsideValue(0)
otsu_filter.SetOutsideValue(1)
image_otsu = otsu_filter.Execute(image);

# Connected component analysis

cca = sitk.ConnectedComponentImageFilter()
image_cca = cca.Execute(image_otsu);

# Statistics of the labels

stats = sitk.LabelShapeStatisticsImageFilter()
relabel = sitk.RelabelComponentImageFilter()
img_relabel = relabel.Execute(image_cca)
img_stats = stats.Execute(img_relabel);
change = sitk.ChangeLabelImageFilter()
k = 0

for i in stats.GetLabels():
#    #print("Label: {0} -> Size: {1}".format(stats.GetLabels(i),stats.GetPhysicalSize(i)))

# Setting centroid of interested region to be within a particular threshold
    if(stats.GetCentroid(i)[0]>(w/2)-(w/6) and stats.GetCentroid(i)[0]<(w/2)+(w/6) and stats.GetCentroid(i)[1]>(h/2)-(h/6) and stats.GetCentroid(i)[1]<(h/2)+(h/6) and i<10):
        print("Size of the interested region {0} :".format(i),stats.GetPhysicalSize(i))
        print("Centroid of interested region {0} :".format(i),stats.GetCentroid(i))
        k = k + 1
    else:
#        print(i)
#        print(i,stats.GetCentroid(i))
        img_relabel = change.Execute(img_relabel,dict([(i,0.0)]))
        
print("Number of interested regions : ",k)    
#    print(i,stats.GetPhysicalSize(i))

#overlay = sitk.LabelOverlayImageFilter()
#image_overlay = overlay.Execute(image_cca,)

#mask = sitk.LabelMapMaskImageFilter()
#img_mask = mask.Execute(img_stats,image)

_, contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
for i, c in enumerate(contours):
    # Calculate the area of each contour
    area = cv.contourArea(c);
#    print (area)
    # Ignore contours that are too small or too large
    if area < 1e3 or 1e5 < area:
        continue
    cnt = contours[i];
    # Draw rectangle around the interested region
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    im = cv.drawContours(img_for_hsv,[box],0,(0,0,255),2)
    # Draw each contour only for visualisation purposes
#    cv.drawContours(img_for_hsv, contours, i, (0, 0, 255), 2);
    # Find the orientation of each shape
    getOrientation(c, img_for_hsv)
cv.imshow('output', img_for_hsv)
cv.waitKey(10000)
# RelabelComponentImageFilter reallots labels such that the largest size object gets #1 allotted and so on

# Writing the result

writer = sitk.ImageFileWriter()
writer.SetFileName( sys.argv[2] )
writer.Execute( image_cca)
#writer.SetFilename( sys.argv[4])
#writer.Execute(mask)
cv.imwrite('mask1.jpg',mask);
cv.imwrite('After PCA.jpg',img_for_hsv)
cv.imwrite('hsvchannel.jpg',mask2)
print(mask.dtype)
#cv.imshow('HSV Image',mask);
#cv.waitKey(10000);
#cv.destroyAllWindows();