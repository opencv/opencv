import cv2
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Color correction sample")
    parser.add_argument("--input", default="opencv_extra/testdata/cv/mcc/mcc_ccm_test.jpg",
                        help="Path of the image file to process")
    parser.add_argument("--colors", default="samples/data/ccm_test_data.txt",
                        help="Path to the txt file containing color values")
    args = parser.parse_args()

    # Read the input image
    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if image is None:
        print("Invalid Image!")
        return 1

    # Read color values from file (expecting 24 lines with three float numbers per line)
    try:
        with open(args.colors, 'r') as infile:
            lines = infile.readlines()
    except Exception:
        print("Failed to open color values file!")
        return 1

    colors = []
    for i in range(24):
        # Remove whitespace and split the line
        parts = lines[i].strip().split()
        if len(parts) >= 3:
            # Parse three float values
            r, g, b = map(float, parts[:3])
            colors.append([r, g, b])

    # Convert list to a NumPy array with shape (24, 1, 3) and type float64
    src = np.array(colors, dtype=np.float64).reshape(24, 1, 3)

    # Create and compute the Color Correction Model
    model1 = cv2.ccm.ColorCorrectionModel(src, cv2.ccm.COLORCHECKER_Macbeth)
    model1.computeCCM()
    ccm = model1.getCCM()
    print("ccm", ccm)
    loss = model1.getLoss()
    print("loss", loss)

    # Save the CCM matrix and loss using OpenCV FileStorage
    fs = cv2.FileStorage("ccm_output.yaml", cv2.FILE_STORAGE_WRITE)
    fs.write("ccm", ccm)
    fs.write("loss", loss)
    fs.release()

    # Convert image from BGR to RGB, then to float64 and normalize to [0, 1]
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float64) / 255.0

    # Apply color correction inference
    calibratedImage = model1.infer(img_rgb)
    out_ = calibratedImage * 255.0
    out_ = np.clip(out_, 0, 255).astype(np.uint8)

    # Convert back to BGR for saving
    out_img = cv2.cvtColor(out_, cv2.COLOR_RGB2BGR)

    # Construct output file name from the input image path
    filename = os.path.basename(args.input)
    baseName, ext = os.path.splitext(filename)
    calibratedFilePath = f"{baseName}.calibrated{ext}"
    cv2.imwrite(calibratedFilePath, out_img)

if __name__ == '__main__':
    main()
