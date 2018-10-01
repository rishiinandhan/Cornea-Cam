# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 17:38:14 2018

@author: rishiinandhan
"""

# Header files
from __future__ import print_function

import SimpleITK as sitk
import sys
import numpy as np
from PIL import Image

# Check minimum number of arguments in command line

if len( sys.argv ) < 3:
  print("Usage: Python InputImage OutputImage");
  sys.exit( 1 )

# Read the image
  
reader = sitk.ImageFileReader()
reader.SetFileName( sys.argv[1] )
image = reader.Execute();
reader.SetFileName( sys.argv[3])
image2 = reader.Execute();
data = sitk.GetArrayFromImage(image);
data2 = sitk.GetArrayFromImage(image2)
print( data.shape)
print(data2.shape)
print(type(data[1,0,0]))
shape = data.shape
#for i in range(0,shape[0]):
#    for j in range(0,shape[1]):
#        if(data[i,j,2]>50):
#            data[i,j,:]=data2[i,j,:]

for i in range(0,shape[0]):
    for j in range(0,shape[1]):
        data[i,j,1] = data[i,j,1]//2 + data2[i,j,1]//2;

data[:,:,2] = data2[:,:,2]            
data[:,:,0] = 0;
#data[:,:,1] = 0;
data[:,:,2] = 0;
#data[:,:,2] = 0;
#data[:,:,0] = 0;
#print(data)
dat =  data;
print((dat.shape))
#print(dat);
data_g = dat
#print(data_g);
print(type(data_g[1,0,0]))
#print(data_g.shape)
#data_g = data[:,:,1]
im = Image.fromarray(data_g);
im.save(sys.argv[2])
#mpl.image.imsave(sys.argv[2],data[:,:,1])
#image = sitk.GetImageFromArray(int(1/2.*data[:,:,1]))
#
#writer = sitk.ImageFileWriter()
#writer.SetFileName( sys.argv[2] )
#writer.Execute( image )