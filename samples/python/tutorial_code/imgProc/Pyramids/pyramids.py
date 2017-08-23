import sys
import cv2


def main(argv):
    print("""
    Zoom In-Out demo
    ------------------
    * [i] -> Zoom [i]n
    * [o] -> Zoom [o]ut
    * [ESC] -> Close program
    """)
    ## [load]
    filename = argv[0] if len(argv) > 0 else "../data/chicky_512.png"

    # Load the image
    src = cv2.imread(filename)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: pyramids.py [image_name -- default ../data/chicky_512.png] \n')
        return -1
    ## [load]
    ## [loop]
    while 1:
        rows, cols, _channels = map(int, src.shape)
        ## [show_image]
        cv2.imshow('Pyramids Demo', src)
        ## [show_image]
        k = cv2.waitKey(0)

        if k == 27:
            break
            ## [pyrup]
        elif chr(k) == 'i':
            src = cv2.pyrUp(src, dstsize=(2 * cols, 2 * rows))
            print ('** Zoom In: Image x 2')
            ## [pyrup]
            ## [pyrdown]
        elif chr(k) == 'o':
            src = cv2.pyrDown(src, dstsize=(cols // 2, rows // 2))
            print ('** Zoom Out: Image / 2')
            ## [pyrdown]
    ## [loop]

    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
