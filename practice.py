import cv2
import numpy as np
import math


def sunidhi_count_nonzero(img):
    count=0;
    for habi in img:
        for j in habi: 
            if j!=0 :
                count +=1
    return count

def sunidhi_rectangle(img,pt1,pt2,color,thickness):
    x1,y1=pt1
    x2,y2=pt2

    if thickness==-1:
        img[y1:y2,x1:x2]=color
    else:
        img[y1:y1+thickness, x1:x2] = color
        img[y2:y2-thickness,x1:x2] = color

        #wrong
        #img[x1:x1+thickness,y1:y2] = color #first index always height,then width
        #img[x2:x2-thickness,y1:y2] = color
        img[y1:y2, x1:x1+thickness] = color
        img[y1:y2, x2-thickness:x2] = color

def gausskernel(size,sigma) :
    gaskernel=np.zeros(size,size)
    norm=0
    padding=(gaskernel.shape[0]-1) // 2
    for x in range(-padding,padding+1):
        for y in range(-padding,padding+1):
            c=1/(2*3.1416*sigma*sigma)
            gaskernel[x+padding,y+padding] = c*math.exp(-(x**2 + y**2) /(2 * sigma ** 2))
            norm+=gaskernel[x+padding, y+padding]
    return gaskernel/norm

def gaussblur(img,kernel_size,sigma):
    gaussiankernel=gausskernel(kernel_size,sigma)
    image_h, image_w = img.shape

    padding_x= (kernel_size-1) //2
    padding_y=(kernel_size-1) //2

    paddedimg=cv2.copyMakeBorder(img,padding_y,padding_y,padding_x,padding_x,cv2.BORDER_REFLECT)
    
    output_image_h= image_h + kernel_size -1
    output_image_w  = image_w + kernel_size -1

    gausoutput= np.zeros(output_image_h,output_image_w)

    for x in range(padding_x,output_image_h - padding_x):
        for y in range(padding_y,output_image_w-padding_y):
            temp = 0 
            for i in range (-padding_x,padding_x+1):
                for j in range (-padding_y,padding_y+1):
                    temp += paddedimg[x-i,y-j] * gaussiankernel[i+padding_x, j+ padding_y]
            gausoutput[x,y] = temp
    
    gausoutput=gausoutput[padding_y:padding_y+image_h,padding_x+padding_x+image_w]
    gausoutput = cv2.normalize(gausoutput,None,0,255,cv2.NORM_MINMAX)
    return gausoutput.astype(np.uint8)

def my_adaptive_threshold(img, max_val,method,threshold_type,block_size,C):
    if block_size%2==0:
        return ValueError("hobe na")
    
    image_h,image_w=img.shape

    output=np.zeros_like(img,dtype=np.uint8)

    padding=block_size//2

    padded=cv2.copyMakeBorder(img,padding,padding,padding,padding,cv2.BORDER_REFLECT)


    for y in range(image_h):
        for x in range(image_w):
            neighborhodd= padded[y:y+block_size,x:x+block_size]

            if method=='mean':
                threshold=np.mean(neighborhodd)-C
            elif method=='gaussian':
                gaussinakernl=cv2.getGaussianKernel(block_size,-1)
                gaussinakernl=gaussinakernl*gaussinakernl.T
                threshold=np.sum(neighborhodd*gaussinakernl)-C
            else:
                raise ValueError("mehtod not good")
            
            if threshold_type=='binary' :
                output[y,x]=max_val if img[y,x]>threshold else 0
            elif threshold_type=='binary_inv':
                output[y,x]=0 if img[y,x]>threshold else max_val
            else:
                raise ValueError("lfalafl")
    return output

def median_filter(img,kernel_size):
    image_h,image_w=img.shape

    padding_x=kernel_size-1 //2
    padding_y=kernel_size-1 //2

    padded=cv2.copyMakeBorder(img,padding_y,padding_y,padding_x,padding_x,cv2.BORDER_REFLECT)

    median_output=np.zeros_like(img,dtype=np.uint8)

    median_index=(kernel_size*kernel_size) //2

    for x in range(padding_x,image_h+padding_x):
        for y in range(padding_y,image_w+padding_y):
            neighborhoodf=[]
            for i in range(-padding_x,padding_x+1):
                for j in range(-padding_y,padding_y+1):
                    neighborhoodf.append(padded[x+i,y+j])
            neighborhoodf.sort()
            median_output[x-padding_x,y-padding_y] = neighborhoodf[median_index]

    return median_output

def my_dilate(img,kernel,iterations=1):
    ....

    dilated_img=np.zeros_like(img)

    for _ in range (iterations):
        for x in range (padd):
            for y in range():
                neighbordhood=padded[x-padding_x:padding_x+1,y-padding_y:y+padding_y+1]
                dilated_img[x-padding_x,y-padding_y]=np.max(neighbordhood*kernel)

            
            make pad again
    return dilated_img