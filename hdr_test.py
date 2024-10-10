import cv2 as cv
import numpy as np

# Loading exposure images into a list
img_fn = ["img1.png", "img2.png", "img3.png"]  # Image filenames
img_list = [cv.imread(fn) for fn in img_fn]  # Load images using OpenCV

# Check if any image failed to load
if any(img is None for img in img_list):
    for i, img in enumerate(img_list):
        if img is None:
            print(f"Failed to load image: {img_fn[i]}")
    exit()  # Exit if any image fails to load
else:
    for i, img in enumerate(img_list):
        print(f"Successfully loaded image: {img_fn[i]}, Dimensions: {img.shape}")

# Exposure times for each image (in seconds)
exposure_times = np.array([8.0, 8.0, 15.0], dtype=np.float32)

# Merge exposures using Robertson's method
merge_robertson = cv.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())

# Check for NaN values and replace them with zeros
if np.isnan(hdr_robertson).any():
    print("HDR result contains NaN values. Replacing NaNs with zeros.")
    hdr_robertson = np.nan_to_num(hdr_robertson)

# Tonemap the HDR image using TonemapReinhard, allowing control of saturation and intensity (bias)
tonemap_reinhard = cv.createTonemapReinhard(gamma=0.8, intensity=-1.0, light_adapt=0.7, color_adapt=0.5)
ldr_reinhard = tonemap_reinhard.process(hdr_robertson.copy())

# Convert the result to 8-bit to save/display
ldr_reinhard_8bit = np.clip(ldr_reinhard * 255, 0, 255).astype('uint8')

# Save the result
cv.imwrite("ldr_robertson_reinhard.jpg", ldr_reinhard_8bit)
print("HDR processing complete. Output saved as 'ldr_robertson_reinhard.jpg'.")
