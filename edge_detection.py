# Simple edge detection script
# This script loads an image, converts it to grayscale, and applies Canny edge detection to find edges.
# The result is displayed in a window.

import cv2

i = cv2.imread('img.jpg')  # Load the image from file
g = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale (removes colors)
e = cv2.Canny(g, 100, 200)  # Apply Canny edge detection (finds edges with thresholds 100 and 200)
cv2.imshow('Edges', e)  # Display the detected edges in a window called 'Edges'
cv2.waitKey(0)  # Wait until user presses a key to close the window
cv2.destroyAllWindows()  # Close all open windows
