#!/usr/bin/env python

import sys, os, os.path, glob, math, cv2
from datetime import datetime
from optparse import OptionParser

def parse(ipath, f):
    bbs = []
    path = None
    for l in f:
        box = None
        if l.startswith("Bounding box"):
            b = [x.strip() for x in l.split(":")[1].split("-")]
            c = [x[1:-1].split(",") for x in b]
            d = [int(x) for x in sum(c, [])]
            bbs.append(d)

        if l.startswith("Image filename"):
            path = os.path.join(os.path.join(ipath, ".."), l.split('"')[-2])

    return (path, bbs)

def adjust(box, tb, lr):

    mix = int(round(box[0] - lr))
    miy = int(round(box[1] - tb))

    max = int(round(box[2] + lr))
    may = int(round(box[3] + tb))

    return [mix, miy, max, may]

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", metavar="DIRECTORY", type="string",
                       help="path to Inria train data folder")

    parser.add_option("-o", "--output", dest="output", metavar="DIRECTORY", type="string",
                       help="path to store data", default=".")

    parser.add_option("-t", "--target", dest="target", type="string", help="should be train or test", default="train")

    (options, args) = parser.parse_args()
    if not options.input:
        parser.error("Inria data folder required")

    if options.target not in ["train", "test"]:
        parser.error("dataset should contain train or test data")

    octaves = [-1, 0, 1, 2]

    path = os.path.join(options.output, datetime.now().strftime("rescaled-" + options.target + "-%Y-%m-%d-%H-%M-%S"))
    os.mkdir(path)

    neg_path = os.path.join(path, "neg")
    os.mkdir(neg_path)

    pos_path = os.path.join(path, "pos")
    os.mkdir(pos_path)

    print "rescaled Inria training data stored into", path, "\nprocessing",
    for each in octaves:
        octave = 2**each

        whole_mod_w = int(64 * octave) + 2 * int(20 * octave)
        whole_mod_h = int(128 * octave) + 2 * int(20 * octave)

        cpos_path = os.path.join(pos_path, "octave_%d" % each)
        os.mkdir(cpos_path)
        idx = 0

        gl = glob.iglob(os.path.join(options.input, "annotations/*.txt"))
        for image, boxes in [parse(options.input, open(__p)) for __p in gl]:
            for box in boxes:
                height = box[3] - box[1]
                scale = height / float(96)

                mat = cv2.imread(image)
                mat_h, mat_w, _ = mat.shape

                rel_scale = scale / octave

                d_w = whole_mod_w * rel_scale
                d_h = whole_mod_h * rel_scale

                top_bottom_border = (d_h - (box[3] - box[1])) / 2.0
                left_right_border = (d_w - (box[2] - box[0])) / 2.0

                box = adjust(box, top_bottom_border, left_right_border)
                inner = [max(0, box[0]), max(0, box[1]), min(mat_w, box[2]), min(mat_h, box[3]) ]

                cropped = mat[inner[1]:inner[3], inner[0]:inner[2], :]

                top     = int(max(0, 0 - box[1]))
                bottom  = int(max(0, box[3] - mat_h))
                left    = int(max(0, 0 - box[0]))
                right   = int(max(0, box[2] - mat_w))
                cropped = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_REPLICATE)
                resized = sft.resize_sample(cropped, whole_mod_w, whole_mod_h)

                out_name = ".png"
                if round(math.log(scale)/math.log(2)) < each:
                    out_name = "_upscaled" + out_name

                cv2.imwrite(os.path.join(cpos_path, "sample_%d" % idx + out_name), resized)

                flipped = cv2.flip(resized, 1)
                cv2.imwrite(os.path.join(cpos_path, "sample_%d" % idx + "_mirror" + out_name), flipped)
                idx = idx + 1
                print "." ,
                sys.stdout.flush()

        idx = 0
        cneg_path = os.path.join(neg_path, "octave_%d" % each)
        os.mkdir(cneg_path)

        for each in [__n for __n in glob.iglob(os.path.join(options.input, "neg/*.*"))]:
            img = cv2.imread(each)
            min_shape = (1.5 * whole_mod_h, 1.5 * whole_mod_w)

            if (img.shape[1] <= min_shape[1]) or (img.shape[0] <= min_shape[0]):
                out_name = "negative_sample_%i_resized.png" % idx

                ratio = float(img.shape[1]) / img.shape[0]

                if (img.shape[1] <= min_shape[1]):
                    resized_size = (int(min_shape[1]), int(min_shape[1] / ratio))

                if (img.shape[0] <= min_shape[0]):
                    resized_size = (int(min_shape[0] * ratio), int(min_shape[0]))

                img = sft.resize_sample(img, resized_size[0], resized_size[1])
            else:
                out_name = "negative_sample_%i.png" % idx

            cv2.imwrite(os.path.join(cneg_path, out_name), img)
            idx = idx + 1
            print "." ,
            sys.stdout.flush()
