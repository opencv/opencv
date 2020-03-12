from __future__ import print_function
import cv2 as cv
import numpy as np 
import sys,getopt
from glob import glob

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

#Input From the camera is to be implemented

#Output Filename and It's extension 
o_f_name=''
o_f_ext=''

#Default values of input parameters
inputfile = '../../../opencv_extra/testdata/cv/qrcode/multiple/*_qrcodes.png'
outputfile = 'qr_code.png'
multi='True'
detect='False'
#save_detection ='False' 
#save_all='False'


helper="""
This program detects the QR-codes input images using OpenCV Library.
Usage: python qrcodesample.py [params]
	-h = print help messages 
	-i, --in (default input file path is 'opencv_extra/testdata/cv/qrcode/multiple/*_qrcodes.png')= input image path  
	-d, --detect (value True/False default value is False) = detect QR code only (skip decoding) 
	-m, --multi (value True/False default value is True) = use detect for multiple qr-codes 
	-o, --out (default output filename is qr_code.png) = path to result file 
"""
def getQRModeString():
	msg1="multi" if(multi=='True') else ""
	msg2="detector" if(detect=='True') else "decoder"
	msg="QR %s %s"%(msg1,msg2) 
	return msg
	
def drawFPS(result,fps):
	message='%.2f FPS(%s)'%(fps,getQRModeString())
	cv.putText(result,message,(20,20),1,cv.FONT_HERSHEY_DUPLEX,(0,0,255))
	
def drawQRCodeContours(image,cnt):
	if(cnt.size!=0):
		rows,cols,channels=image.shape
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
	if(multi=='False'):
		if(detect=='False'):
			decode_info,points,straight_qrcode=qrCode.detectAndDecode(inputimg)
			dec_info=decode_info
		else:
			retval,points=qrCode.detect(inputimg)
			dec_info=[]
	else:
		if(detect=='False'):
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
	for i in range(count):
		timer.start()
		points,decode_info=runQR(qrCode,inputimg)
		timer.stop()
	fps = count/timer.getTimeSec()
	print("FPS: ", fps)
	result = inputimg
	drawQRCodeResults(result,points,decode_info,fps)
	cv.imshow("QR",result)
	cv.waitKey(1)
	if(outputfile!=''):
		outfile = o_f_name+o_f_ext;
		print ("Saving Result: ", outfile)
		cv.imwrite(outfile,result)
		
	print("Press any key to exit ...")
	cv.waitKey(0)
	print("Exit")
	
	
	
	
	
def main(argv):
   global o_f_name
   global o_f_ext
   global outputfile
   global inputfile
   global multi
   global detect
   #global save_all
   #global save_detection
   try:
      opts, args = getopt.getopt(argv,"hi:o:m:d:",["in=","out=","multi=","detect="])
   except getopt.GetoptError:
      print(helper)
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print(helper)
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
      elif opt in ("-m","--multi"):
		  multi = arg
      elif opt in ("-d",'--detect'):
		  detect = arg
			  
   if(outputfile!=''):
      index=outputfile.rfind('.')
      if(index!=-1):
         o_f_name=outputfile[:index]
         o_f_ext=outputfile[index:]
      else:
         o_f_name=outputfile
         o_f_ext=".png"
   for fn in glob(inputfile):
	   DetectQRFrmImage(fn)
	   

if __name__== '__main__':
	main(sys.argv[1:])


