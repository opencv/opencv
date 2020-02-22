"""
This is a basic Python implementation of Otsu's Thresholding.
The purpose of this implementation is to impart a better
understanding of how Otsu's Thresholding works.
Tested on Python 3.
"""

#Importing Required Libraries
import argparse
import cv2 as cv
import numpy as np

def otsu_thresholding(input_image):
    
    image = input_image.copy()
    rows, cols = image.shape
    number_of_pixels = rows*cols
    
    minimum_variance = np.inf
    threshold_value = 0
          
    ###### Following steps are done to avoid redundant computation #####
    
    #To store the occurence of each intensity
    occurence = [0] * 256 
    #To store the sum intensities upto a specific intensity
    boundary_count = [0] * 256 #
        
    # Saves the corresponding intensity of each pixel on each index
    for i in range(rows):
        for j in range(cols):
            occurence[image[i][j]] += 1

    # Sum of the previous intensities upto a specific intensity for which 
    # the within class variance will be calculated
    boundary_count[0] = occurence[0]
    for i in range(1, 256):
        #Adding previous count to current count
        boundary_count[i] = occurence[i] + boundary_count[i-1]
    ####################################################################
     
   
    #Finding the lowest class variance by placing a boundary at each intensity
    for boundary in range(0, 256):
        
        thresh_boundary = boundary #Set intensity as class boundary
        
        #Calculated Above
        upper_count = boundary_count[255] - boundary_count[boundary]
        lower_count = boundary_count[boundary] #Calculated Above
        
        if(upper_count != 0 and lower_count != 0): #Avoiding divison by zero
        
    # Computing Weights For Both Classes (Class Count/Total Pixel Count)
            lower_weight = lower_count/number_of_pixels
            upper_weight = upper_count/number_of_pixels
            
    #Computing Weighted Mean For Each Class
            lower_mean = 0.0
            upper_mean = 0.0
            
            #(Intensity x Frequency of Intensity)
            for intensity in range(0, thresh_boundary+1): #Lower Class
                lower_mean += intensity*occurence[intensity] 
                 
            for intensity in range(thresh_boundary, 256): #Upper Class
                upper_mean += intensity*occurence[intensity]
            
            #Sum of Intensities/Count            
            lower_mean = lower_mean/lower_count
            upper_mean = upper_mean/upper_count
                
    #Computing Variance
            lower_variance = 0.0
            upper_variance = 0.0
            
            # ((Intensity Value - Mean)**2) x Frequency of Intensity
            for value in range(0, thresh_boundary+1):
                lower_variance += (value-lower_mean)**2*occurence[value]
            lower_variance = lower_variance/lower_count
            
            for value in range(thresh_boundary, 256):
                upper_variance += ((value-upper_mean)**2)*occurence[value]
            upper_variance = upper_variance/upper_count
            
    #Within Class Variance
            within_classVariance = (lower_variance*lower_weight) + (upper_variance*upper_weight)
            
    #The optimal threshold is set at boundary which gives the minimum variance
            if(within_classVariance < minimum_variance):
                minimum_variance = within_classVariance
                threshold_value = thresh_boundary
            
    print("Threshold was held at: ", threshold_value)  

    #Applying Threshold at Computed Optimal Threshold
    ret, otsu_output_image = cv.threshold(input_image,threshold_value,255,cv.THRESH_BINARY)
    return otsu_output_image
        

#Argument Parser
parser = argparse.ArgumentParser(description='Tutorial Code for Otsu Thresholding')
parser.add_argument('--input', help='Enter path to input image.', default='image.jpg')
args = parser.parse_args()

# Reading Image
read_image = cv.imread(cv.samples.findFile(args.input))
if read_image is None:
    print('Failed to load image: ', args.input)
    exit(0)

read_image = cv.cvtColor(read_image, cv.COLOR_BGR2GRAY)

#Applying Otsu's Thresholding
result = otsu_thresholding(read_image)

#Comparing results with original image
output = np.concatenate((read_image, result), axis = 1)
print("Done Processing...")

cv.imshow("Output", output)
cv.waitKey()
cv.destroyAllWindows()
