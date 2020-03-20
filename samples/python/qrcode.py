from __future__ import print_function
import cv2 as cv
import numpy as np
import sys
from glob import glob
PY3 = sys.version_info[0] == 3
import argparse

if PY3:
    xrange = range

def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#Input From the camera is to be implemented

#Output Filename and It's extension
o_f_name=''
o_f_ext=''

#Default values of input parameters
parser = argparse.ArgumentParser(description='This program detects the QR-codes input images using OpenCV Library.')
parser.add_argument('-i','--input',help="input image path (default input file path is 'opencv_extra/testdata/cv/qrcode/multiple/*_qrcodes.png",default="../../../opencv_extra/testdata/cv/qrcode/multiple/*_qrcodes.png",metavar="")
parser.add_argument('-d','--detect',help="detect QR code only (skip decoding) (value True/False default value is False)",metavar='',type=str2bool,default=False)
parser.add_argument('-m','--multi',help="use detect for multiple qr-codes (value True/False default value is True)",metavar="",type=str2bool,default=True)
parser.add_argument('-o','--out',help="path to result file (default output filename is qr_code.png)",metavar="",default="qr_code.png")
args = parser.parse_args()
print(args)
#save_detection ='False'
#save_all='False'


def getQRModeString():
    msg1="multi" if(args.multi) else ""
    msg2="detector" if(args.detect) else "decoder"
    msg="QR %s %s"%(msg1,msg2)
    return msg

def drawFPS(result,fps):
    message='%.2f FPS(%s)'%(fps,getQRModeString())
    cv.putText(result,message,(20,20),1,cv.FONT_HERSHEY_DUPLEX,(0,0,255))

def drawQRCodeContours(image,cnt):
    if(cnt.size!=0):
        rows,cols,_=image.shape
        show_radius = (2.813*rows)/cols if (rows>cols) else (2.813*cols)/rows
        contour_radius=show_radius*0.4
        cv.drawContours(image,[cnt],0,(0,255,0),int(round(contour_radius)))
        tpl=cnt.reshape((-1,2))
        for x in tuple(tpl.tolist()):
            color=(255,0,0)
            cv.circle(image,tuple(x),int(round(contour_radius)),color,-1)

def drawQRCodeResults(result,points,decode_info,fps):
    n=len(points)
    if(type(decode_info) is str):
        decode_info=[decode_info]
    if(n>0):
        for i in range(n):
            cnt=np.array(points[i]).reshape((-1,1,2)).astype(np.int32)
            drawQRCodeContours(result,cnt)
            print ('QR[',i,']@',cnt.reshape(1,-1).tolist(),": ",end="")
            if(len(decode_info)>i):
                if(decode_info[i]):
                    print("'",decode_info[i],"'")
                else:
                    print("Can't decode QR code")
            else:
                print("Decode information is not available (disabled)")
    else:
        print ("QRCode not  detected!")
    drawFPS(result,fps)

def runQR(qrCode,inputimg):
    if(not args.multi):
        if(not args.detect):
            decode_info,points,_=qrCode.detectAndDecode(inputimg)
            dec_info=decode_info
        else:
            _,points=qrCode.detect(inputimg)
            dec_info=[]
    else:
        if(not args.detect):
            retval,decode_info,points,straight_qrcode=qrCode.detectAndDecodeMulti(inputimg)
            dec_info=decode_info
        else:
            retval,points=qrCode.detectMulti(inputimg)
            dec_info=[]
    if(points is None):
        points=[]
    return points,dec_info

def DetectQRFrmImage(inputfile):
    inputimg = cv.imread(inputfile,cv.IMREAD_COLOR)
    qrCode = cv.QRCodeDetector()
    count = 10
    timer=cv.TickMeter()
    for _ in range(count):
        timer.start()
        points,decode_info=runQR(qrCode,inputimg)
        timer.stop()
    fps = count/timer.getTimeSec()
    print("FPS: ", fps)
    result = inputimg
    drawQRCodeResults(result,points,decode_info,fps)
    cv.imshow("QR",result)
    cv.waitKey(1)
    if(args.out!=''):
        outfile = o_f_name+o_f_ext
        print ("Saving Result: ", outfile)
        cv.imwrite(outfile,result)

    print("Press any key to exit ...")
    cv.waitKey(0)
    print("Exit")

def main(argv):
    global o_f_name
    global o_f_ext
    if(args.out!=''):
        index=args.out.rfind('.')
        if(index!=-1):
            o_f_name=args.out[:index]
            o_f_ext=args.out[index:]
        else:
            o_f_name=args.out
            o_f_ext=".png"
    for fn in glob(args.input):
        DetectQRFrmImage(fn)


if __name__== '__main__':
    main(sys.argv[1:])
