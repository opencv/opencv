import time
import sys
import cv2 as cv
import numpy as np

class Settings:
    # Pattern
    NOT_EXISTING = 0
    CHESSBOARD = 1
    CIRCLES_GRID = 2
    ASYMMETRIC_CIRCLES_GRID = 3

    # InputType
    INVALID = 0
    CAMERA = 1
    VIDEO_FILE = 2
    IMAGE_LIST = 3

    def __init__(self):
        self.goodInput = False
    def write(self, fs):
        fs.write('Settings', '{')
        fs.write('BoardSize_Width', self.boardSize.width)
        fs.write('BoardSize_Height', self.boardSize.height)
        fs.write('Square_Size', self.squareSize)
        fs.write('Calibrate_Pattern', self.patternToUse)
        fs.write('Calibrate_FixAspectRatio', self.nrFrames)
        fs.write('Calibrate_AssumeZeroTangentialDistortion', self.calibZeroTangentDist)
        fs.write('Write_DetectedFeaturePoints', self.writePoints)
        fs.write('Write_extrinsicParameters', self.writeExtrinsics)
        fs.write('Write_gridPoints', self.writeGrid)
        fs.write('Write_outputFileName', self.outputFileName)
        fs.write('Show_UndistortedImage', self.showUndistorsed)
        fs.write('Input_FlipAroundHorizontalAxis', self.flipVertical)
        fs.write('Input_Delay', self.delay)
        fs.write('Input', self.input)
        fs.write('Settings', '}')
    def read (self, node):
        self.boardSize.width = int(node.getNode('BoardSize_Width').real())
        self.boardSize.height = int(node.getNode('BoardSize_Height').real())
        self.patternToUse = node.getNode('Calibrate_Pattern').string()
        self.squareSize = node.getNode('Square_Size').real()
        self.nrFrames = int(node.getNode('Calibrate_FixAspectRatio').real())
        self.aspectRatio = node.getNode('Calibrate_FixAspectRatio').real()
        self.writePoints = bool(node.getNode('Write_DetectedFeaturePoints').real())
        self.writeExtrinsics = bool(node.getNode('Write_extrinsicParameters').real())
        self.writeGrid = bool(node.getNode('Write_gridPoints'))
        self.outputFileName = node.getNode('Write_outputFileName').string()
        self.calibZeroTangentDist = bool(node.getNode('Calibrate_AssumeZeroTangentialDistortion').real())
        self.calibFixPrincipalPoint = bool(node.getNode('Calibrate_FixPrincipalPointAtTheCenter').real())
        self.useFisheye = bool(node.getNode('Calibrate_UseFisheyeModel').real())
        self.flipVertical = bool(node.getNode('Input_FlipAroundHorizontalAxis').real())
        self.showUndistorsed = bool(node.getNode('Show_UndistortedImage').real())
        self.input = node.getNode('Input').string()
        self.delay = int(node.getNode('Input_Delay').real())
        self.fixK1 = bool(node.getNode('Fix_K1').real())
        self.fixK2 = bool(node.getNode('Fix_K2').real())
        self.fixK3 = bool(node.getNode('Fix_K3').real())
        self.fixK4 = bool(node.getNode('Fix_K4').real())
        self.fixK5 = bool(node.getNode('Fix_K5').real())
        validate(self)
    def validate(self):
        self.goodInput = True
        if self.boardSize.width <= 0 or self.boardSize.height <= 0:
            print('Invalid Board size:', self.boardSize.width, self.boardSize.height, file=sys.stderr)
            self.goodInput = False
        if self.squareSize <= 10e-6:
            print('Invalid square size', self.squareSize, file=sys.stderr)
            self.goodInput = False
        if nrFrames <= 0:
            print('Invalid number of frames', self.nrFrames, file=sys.stderr)
            self.goodInput = False
        if self.input.empty():
            self.inputType = INVALID
        else:
            if self.input[0].isdigit():
                self.cameraID = int(self.input)
                self.inputType = CAMERA
            else:
                if isListOfImages(self.input) and readStringList(self.input, imageList):
                    self.inputType = IMAGE_LIST
                    self.nrFrames = self.nrFrames if self.nrFrames < self.imageList.size() else self.imageList.size()
                else:
                    self.inputType = VIDEO_FILE
            if self.inputType == CAMERA
                self.inputCapture.open(self.cameraID)
            if self.inputType == VIDEO_FILE:
                self.inputCapture.open(self.input)
            if self.inputType == IMAGE_LIST and not self.inputCapture.isOpened():
                self.inputType = INVALID
        if self.inputType == INVALID:
            print('Input does not exist:', self.input, file=sys.stderr)
            self.goodInput = False

        self.flag = 0;
        if calibFixPrincipalPoint: self.flag |= cv.CALIB_FIX_PRINCIPAL_POINT
        if calibZeroTangentDist:   self.flag |= cv.CALIB_ZERO_TANGENT_DIST
        if aspectRatio:            self.flag |= cv.CALIB_FIX_ASPECT_RATIO
        if self.fixK1:             self.flag |= cv.CALIB_FIX_K1
        if self.fixK2:             self.flag |= cv.CALIB_FIX_K2
        if self.fixK3:             self.flag |= cv.CALIB_FIX_K3
        if self.fixK4:             self.flag |= cv.CALIB_FIX_K4
        if self.fixK5:             self.flag |= cv.CALIB_FIX_K5
        if self.useFisheye:
            self.flag = fisheye.CALIB_FIX_SKEW | fisheye.CALIB_RECOMPUTE_EXTRINSIC 
            if self.fixK1: self.flag |= fisheye.CALIB_FIX_K1
            if self.fixK2: self.flag |= fisheye.CALIB_FIX_K2
            if self.fixK3: self.flag |= fisheye.CALIB_FIX_K3
            if self.fixK4: self.flag |= fisheye.CALIB_FIX_K4
            if self.fixK5: self.flag |= fisheye.CALIB_FIX_K5
            if self.calibFixPrincipalPoint: self.flag |= fisheye.CALIB_FIX_PRINCIPAL_POINT
        self.calibrationPattern = NOT_EXISTING
        if self.patternToUse == 'CHESSBOARD':
            self.calibrationPattern = CHESSBOARD
        if self.patternToUse == 'CIRCLES_GRID':
            self.calibrationPattern = CIRCLE_GRID
        if self.patternToUse == 'ASYMMETRIC_CIRCLES_GRID':
            self.calibrationPattern = ASYMMETRIC_CIRCLES_GRID
        if self.calibrationPattern == NOT_EXISTING:
            print(' Camera calibration mode does not exist:', self.patternToUse, file=sys.stderr)
            self.goodInput = False
        self.atImageList = 0
    def nextImage(self):
        if self.inputCapture.isOpened():
            self.inputCapture.read(result)
        else if self.atImageList < self.imageList.size():
            result = imread(self.imageList[self.atImageList++], cv.IMREAD_COLOR)
        return result
    def readStringList(filename, l):
        l.clear()
        fs = FileStorage(filename, cv.FileStorage_READ)
        if not fs.isOpened():
            return False
        n = fs.getFirstTopLevelNode()
        if n.type() != cv.FileNode_SEQ:
            return False
        for i in range(n.size()):
            l.append(n.at(i).string())
        return True
    def isListOfImages(filename):
        return not (s.find('.xml') != -1 and s.find('.yaml') != -1 and s.find('.yml') != -1 and s.find('.json') != -1)

if __name__ == '__main__':

