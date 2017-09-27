#! /usr/bin/env python
# Author: Sundar Ram Swaminathan
import os
import sys
import csv
import cv2
import glob
import numpy as np
from operator import itemgetter, attrgetter, methodcaller

#function to sort the slopes in increasing order
def sort_slope(slope1,slope2):
    return cmp(slope1[0] , slope2[0])

if __name__ == "__main__":

    #To show the lanes detected in the images
    cv2.namedWindow('Lane Markers', cv2.WINDOW_NORMAL)
    
    #importing all images in the folder
    imgs = glob.glob("images/*.png")

    #Target directories to store the images after detecting the lanes
    targetDir = "C:\Python27\Programming Evaluation\lanes"
##    targetDir2 = "C:\Python27\Programming Evaluation\edges"
##    targetDir3 = "C:\Python27\Programming Evaluation\thresh"
    
    intercepts = []

    #Looping through each image in the folder

    for fname in imgs:
        # Load image and prepare output image
        img = cv2.imread(fname)
        
        #Color to grayscale conversion 
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        height, width,depth = img.shape

        #Removing top half of the image cause to reduce search space
        gray[0:height/2, 0:width] = 0

        #Smoothing the image to reduce noise
        blur = cv2.GaussianBlur(gray,(5,5),0)

        # Binary Thresholding the image               // NOT PERFECT - IMPROVISE
##        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
##            cv2.THRESH_BINARY,11,2)
        #ret,thresh = cv2.threshold(blur,127,255,  cv2.THRESH_BINARY)

        #Canny Edge detection to detect the lanes (minVal and maxVal chosen by trial and error)
        edges = cv2.Canny(blur,50,100,apertureSize = 3)
        
        #Finding contours along the edges 
        contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # Remove contours whose area less than a threshold
        allIndex = []
        for index in range(len(contours)):
            area = cv2.contourArea(contours[index])
            if area < 50: allIndex.append(index)
        allIndex.sort(reverse=True)
        for index in allIndex: contours.pop(index)
        
        # Fitting a line to each good contour
        lines = []
        for cnt in contours:
             # Approximating contour to be in the shape of a line
             approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
             if len(approx)==2:
                 
                 rows,cols = img.shape[:2]
                 # If a line contour is found, fit a straight line onto that contour
                 [vx,vy,x,y] = cv2.fitLine(cnt, cv2.cv.CV_DIST_L2,0,0.01,0.01)
                 lefty = int((-x*vy/vx) + y)
                 righty = int(((cols-x)*vy/vx)+y)
                 lines.append([cols-1,righty,0,lefty])
                 
                 #cv2.line(img,(cols-1,righty),(0,lefty),(0,255,255),2)
                 
                 #cv2.drawContours(img, [cnt], 0, (0,255,0), 5)

        # Now, the problem is I was able to find multiple lines in the image, we have to  choose which line is the correct line that detects the left and lane markers
        # So, I find the slopes of each line. the slopes of the left lane are negative value and that of the right lanes are positive.
        # The slope of the lines on the farthest left are more negative and the slope increases as we move towards the right side of the image
        # So I arrange all the slopes in ascending order and choose the two slopes that are nearest to the zero corssing. The one to the left of the zero corssing is the left lane marker and the one on the right side of the zero crossing is the righ lane marker
        # 
        slopes = []
        # finding the slopes
        for line in lines:
            delta_x = line[1] - line[3]
            delta_y = line[0] - line[2]
            #Storing the slope and the line coordinates in a list to easily map the slope to the line
            slopes.append([float(delta_y) / float(delta_x),line])
            #Sorting slopes in ascending order
            slopes.sort(sort_slope)

        
        # Color = yellow
        color = (0,255,255)

        # Drawing lane markers
        # Finding the intercepts using equation of a line : (y - y1) = m(x - x1)
        if len(slopes)==0:
            left_x = 'None'
            right_x = 'None'
        elif len(slopes) == 1:
            if (slopes[0][0] < 0):
                #left_x = slopes[0][1][2]
                right_x = 'None'
                left_x = (1200 + (slopes[0][1][2]/slopes[0][0]) - slopes[0][1][3]) / (1/slopes[0][0])          
                cv2.line(img,(slopes[0][1][0],slopes[0][1][1]),(slopes[0][1][2],slopes[0][1][3]),color,2)
            else:
                left_x = 'None'
                #right_x = slopes[0][1][0]
                right_x = (1200 + (slopes[0][1][2]/slopes[0][0]) - slopes[0][1][3]) / (1/slopes[0][0])
                cv2.line(img,(slopes[0][1][0],slopes[0][1][1]),(slopes[0][1][2],slopes[0][1][3]),color,2)
          
        elif len(slopes)>1:
            flag = 0 
            for ind in range(len(slopes)):
                if (slopes[ind-1][0] < 0 and slopes[ind][0] > 0):
                    flag = 1
                    #left_x = slopes[ind-1][1][2]
                    #right_x = slopes[ind][1][0]
                    left_x = (1200 + (slopes[ind-1][1][2]/slopes[ind-1][0]) - slopes[ind-1][1][3]) / (1/slopes[ind-1][0])
                    right_x = (1200 + (slopes[ind][1][2]/slopes[ind][0]) - slopes[ind][1][3]) / (1/slopes[ind][0])
                    #Draw Left lane marker
                    cv2.line(img,(slopes[ind-1][1][0],slopes[ind-1][1][1]),(slopes[ind-1][1][2],slopes[ind-1][1][3]),color,2)
                    #Draw Right lane marker
                    cv2.line(img,(slopes[ind][1][0],slopes[ind][1][1]),(slopes[ind][1][2],slopes[ind][1][3]),color,2)
                if flag == 0:
                    
                    neg = [k[0] for k in slopes if k[0] < 0]
                    pos = [k[0] for k in slopes if k[0] > 0]
                    if len(neg)==len(slopes):
                        #left_x = slopes[len(slopes)-1][1][2]
                        left_x = (1200 + (slopes[len(slopes)-1][1][2]/slopes[len(slopes)-1][0]) - slopes[len(slopes)-1][1][3]) / (1/slopes[len(slopes)-1][0])
                        right_x = 'None'
                        cv2.line(img,(slopes[len(slopes)-1][1][0],slopes[len(slopes)-1][1][1]),(slopes[len(slopes)-1][1][2],slopes[len(slopes)-1][1][3]),color,2)
                    elif len(pos)==len(slopes):
                        left_x = 'None'
                        #right_x = slopes[0][1][0]
                        right_x = (1200 + (slopes[0][1][2]/slopes[0][0]) - slopes[0][1][3]) / (1/slopes[0][0])
                        cv2.line(img,(slopes[0][1][0],slopes[0][1][1]),(slopes[0][1][2],slopes[0][1][3]),color,2)
              
            
#            print len(slopes)
        print slopes
        
        if len(contours) == 0:
            print 'NO VALUABLE CONTOURS'
        
        # Sample intercepts
        intercepts.append((os.path.basename(fname), left_x, right_x))

        # Show image
        cv2.imshow('Lane Markers', img)
        #cv2.imshow('thresh', thresh)
        
        #Saving lane detected images into a folder
        cv2.imwrite(os.path.join(targetDir, os.path.basename(fname)),img)

        #Saving the edge detected images into a different folder
        #cv2.imwrite(os.path.join(targetDir2, os.path.basename(fname)),edges)

        #cv2.imwrite(os.path.join(targetDir3, os.path.basename(fname)),thresh)
        key = cv2.waitKey(1000)
        
        if key == 27:
            sys.exit(0)
                
    # CSV output
    with open('intercepts.csv', 'w') as f:
        writer = csv.writer(f)    
        writer.writerows(intercepts)
        
    cv2.destroyAllWindows();
    	
