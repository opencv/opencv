import cv2 as cv
import numpy as np

def main(filename):
    ## [write_animation]
    if filename == "animated_image.png":
        # Create an Animation instance to save
        animation_to_save = cv.Animation()

        # Generate a base image with a specific color
        image = np.full((128, 256, 4), (150, 150, 150, 255), dtype=np.uint8)
        duration = 200
        frames = []
        durations = []

        # Populate frames and durations in the Animation object
        for i in range(10):
            frame = image.copy()
            cv.putText(frame, f"Frame {i}", (30, 80), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 100, 0, 255), 2)
            frames.append(frame)
            durations.append(duration)

        animation_to_save.frames = frames
        animation_to_save.durations = durations

        # Write the animation to file
        cv.imwriteanimation(filename, animation_to_save, [cv.IMWRITE_WEBP_QUALITY, 100])
    ## [write_animation]

    ## [init_animation]
    animation = cv.Animation()
    ## [init_animation]

    ## [read_animation]
    success, animation = cv.imreadanimation(filename)
    if not success:
        print("Failed to load animation frames")
        return
    ## [read_animation]

    ## [show_animation]
    print(" 'press Esc' key to next step")
    escape = 0
    while escape < 1:
        for i, frame in enumerate(animation.frames):
            cv.imshow("Animation", frame)
            cv.moveWindow("Animation", 200, 30)
            key_code = cv.waitKey(animation.durations[i])
            if key_code == 27:  # Exit if 'Esc' key is pressed
                escape = 1
            if escape == 1:
                break
    ## [show_animation]

    ## [init_imagecollection]
    collection1 = cv.ImageCollection(filename, cv.IMREAD_UNCHANGED)
    collection2 = cv.ImageCollection(filename, cv.IMREAD_REDUCED_GRAYSCALE_2 | cv.IMREAD_COLOR_RGB)
    collection3 = cv.ImageCollection(filename, cv.IMREAD_REDUCED_GRAYSCALE_2)
    collection4 = cv.ImageCollection(filename, cv.IMREAD_COLOR_RGB)
    ## [init_imagecollection]

    ## [imreadmulti]
    _, framesBGR = cv.imreadmulti(filename, flags=cv.IMREAD_COLOR_BGR)
    _, framesRGB = cv.imreadmulti(filename, flags=cv.IMREAD_COLOR_RGB)
    ## [imreadmulti]

    ## [check_error]
    if collection1.getStatus():
        print("Failed to initialize ImageCollection")
        import sys
        sys.exit(-1)
    ## [check_error]

    ## [info]
    size = collection1.size()
    width = collection1.getWidth()
    height = collection1.getHeight()
    type_info = collection1.getType()

    print("\nBefore reading the image data the following information can be retrieved")
    print(f"Total Frame count   : {size}")
    print(f"Frame Width  : {width}")
    print(f"Frame Height : {height}")
    print(f"Frame Color Type   : {type_info}")
    ## [info]

    ## [controls]
    idx1 = idx2 = idx3 = 0

    print("\nControls to navigation test:\n"
          "  1: call ImageCollection1.close()\n"
          "  2: call ImageCollection1.releaseCache(5)\n"
          "  3: call ImageCollection.init(...GRAYSCALE_2))\n"
          "  4: call ImageCollection.init(...RGB))\n"
          "  a/d: prev/next idx1\n"
          "  j/l: prev/next idx2\n"
          "  z/c: prev/next idx3\n"
          "  ESC or q: exit")
    ## [controls]

    ## [show_images]
    while True:
        cv.imshow("Image 1", collection1.at(idx1))
        cv.imshow("Image 2", collection2.at(idx2))
        cv.imshow("Image 3", collection3.at(idx3))
        cv.imshow("Image 4", collection4.at(idx1))
        cv.imshow("framesBGR", framesBGR[idx1])
        cv.imshow("framesRGB", framesRGB[idx1])
        cv.imshow("Animation", animation.frames[idx1])

        cv.moveWindow("Image 1", 200, 200)
        cv.moveWindow("Image 2", 500,200)
        cv.moveWindow("Image 3", 700, 200)
        cv.moveWindow("Image 4", 200, 400)
        cv.moveWindow("framesBGR", 500, 400)
        cv.moveWindow("framesRGB", 800, 400)

        key = cv.waitKey(0)

        if key == ord('1'):
            print("collection1 closed")
            collection1.close()
        if key == ord('2'):
            print("collection1 releaseCache(5) called")
            collection1.releaseCache(5)
        if key == ord('3'):
            print("collection1 init called",collection1.size())
            collection1.init(filename, cv.IMREAD_REDUCED_GRAYSCALE_2)
        if key == ord('4'):
            print("collection1 init called",collection1.size())
            collection1.init(filename, cv.IMREAD_COLOR_RGB)
        if key == ord('a'):
            idx1 -= 1
        elif key == ord('d'):
            idx1 += 1
        elif key == ord('j'):
            idx2 -= 1
        elif key == ord('l'):
            idx2 += 1
        elif key == ord('z'):
            idx3 -= 1
        elif key == ord('c'):
            idx3 += 1
        elif key in (ord('q'), 27):  # ESC or q
            break

        idx1 = max(0, min(idx1, collection1.size() - 1))
        idx2 = max(0, min(idx2, collection2.size() - 1))
        idx3 = max(0, min(idx3, collection3.size() - 1))
    ## [show_images]

    ## [cleanup]
    cv.destroyAllWindows()
    ## [cleanup]

if __name__ == "__main__":
    ## [main_call]
    import sys
    print("This is a sample script to test Animation and ImageCollection")
    main(sys.argv[1] if len(sys.argv) > 1 else "animated_image.png")
    ## [main_call]
