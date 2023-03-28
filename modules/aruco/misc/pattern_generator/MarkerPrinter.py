#!/usr/bin/env python3

# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2019, Josh Chien. All rights reserved.

from argparse import ArgumentParser
import numpy as np
from PIL import Image
import io
import warnings
import os
import cairo
from cairosvg import svg2png
import math
import tempfile

def SaveArucoDictBytesList(filePath = "arucoDictBytesList.npz"):
    import numpy as np

    # cv2 is optional dependency
    try:
        import cv2
        from cv2 import aruco

        # Name, Flag
        dictInfo = \
        [
            ("DICT_4X4_1000", aruco.DICT_4X4_1000),
            ("DICT_5X5_1000", aruco.DICT_5X5_1000),
            ("DICT_6X6_1000", aruco.DICT_6X6_1000),
            ("DICT_7X7_1000", aruco.DICT_7X7_1000),
            ("DICT_ARUCO_ORIGINAL", aruco.DICT_ARUCO_ORIGINAL),
            ("DICT_APRILTAG_16h5", aruco.DICT_APRILTAG_16h5),
            ("DICT_APRILTAG_25h9", aruco.DICT_APRILTAG_25h9),
            ("DICT_APRILTAG_36h10", aruco.DICT_APRILTAG_36h10),
            ("DICT_APRILTAG_36h11", aruco.DICT_APRILTAG_36h11),
        ]

        arucoDictBytesList = {}
        for name, flag in dictInfo:
            arucoDict = aruco.getPredefinedDictionary(flag)
            arucoDictBytesList[name] = arucoDict.bytesList

        np.savez_compressed(filePath, **arucoDictBytesList)
        return arucoDictBytesList

    except Exception as e:
        warnings.warn(str(e))
        return None

    return None

class MarkerPrinter:

    debugMode = None # "LINE" "BLOCK"

    # Static Vars
    # SVG https://oreillymedia.github.io/Using_SVG/guide/units.html
    # for PDF and SVG, 1 pixel = 1/72 inch, 1 cm = 1/2.54 inch, 1pixl = 2.54/72 cm, 1cm = 72/2.54 pixels
    ptPerMeter = 72 / 2.54 * 100

    surface = {
            ".SVG": cairo.SVGSurface,
            ".PDF": cairo.PDFSurface,
            ".PS": cairo.PSSurface }

    if (os.path.isfile("arucoDictBytesList.npz")):
        arucoDictBytesList = np.load("arucoDictBytesList.npz")
    else:
        warnings.warn("Missing build-in arucoDictBytesList.npz, generate it again")
        arucoDictBytesList = SaveArucoDictBytesList(filePath = "arucoDictBytesList.npz")

    arucoDictMarkerSize = \
        {
            "DICT_4X4_1000": 4,
            "DICT_5X5_1000": 5,
            "DICT_6X6_1000": 6,
            "DICT_7X7_1000": 7,
            "DICT_ARUCO_ORIGINAL": 5,
            "DICT_APRILTAG_16h5": 4,
            "DICT_APRILTAG_25h9": 5,
            "DICT_APRILTAG_36h10": 6,
            "DICT_APRILTAG_36h11": 6,
        }

    def ArucoBits(dictionary, markerID):
        bytesList = MarkerPrinter.arucoDictBytesList[dictionary][markerID].ravel()
        markerSize = MarkerPrinter.arucoDictMarkerSize[dictionary]

        arucoBits = np.zeros(shape = (markerSize, markerSize), dtype = bool)
        base2List = np.array( [128, 64, 32, 16, 8, 4, 2, 1], dtype = np.uint8)
        currentByteIdx = 0
        currentByte = bytesList[currentByteIdx]
        currentBit = 0
        for row in range(markerSize):
            for col in range(markerSize):
                if(currentByte >= base2List[currentBit]):
                    arucoBits[row, col] = True
                    currentByte -= base2List[currentBit]
                currentBit = currentBit + 1
                if(currentBit == 8):
                    currentByteIdx = currentByteIdx + 1
                    currentByte = bytesList[currentByteIdx]
                    if(8 * (currentByteIdx + 1) > arucoBits.size):
                        currentBit = 8 * (currentByteIdx + 1) - arucoBits.size
                    else:
                        currentBit = 0;
        return arucoBits

    def __DrawBlock(context,
        dictionary = None, markerLength = None, borderBits = 1,
        chessboardSize = (1, 1), squareLength = None, firstMarkerID = 0,
        blockX = 0, blockY = 0, originX = 0, originY = 0, pageBorderX = 0, pageBorderY = 0,
        mode = "CHESS" ):

        if(squareLength is None):
            squareLength = markerLength

        if(markerLength is None):
            markerLength = squareLength

        if((squareLength is None) or (markerLength is None)):
            raise ValueError("lenght is None")

        dawMarkerBlock = False
        if ((mode == "ARUCO") or (mode == "ARUCOGRID")):
            dawMarkerBlock = True
        elif(chessboardSize[1] % 2 == 0):
            dawMarkerBlock = (( blockX % 2 == 0 ) == ( blockY % 2 == 0 ))
        else:
            dawMarkerBlock = (( blockX % 2 == 0 ) != ( blockY % 2 == 0 ))

        if(dawMarkerBlock):
            if (mode != "CHESS"):
                if(dictionary is None):
                    raise ValueError("dictionary is None")

                if (mode == "CHARUCO"):
                    originX = (blockX - originX) * squareLength + (squareLength - markerLength)*0.5 + pageBorderX
                    originY = (blockY - originY) * squareLength + (squareLength - markerLength)*0.5 + pageBorderY
                else:
                    originX = (blockX - originX) * squareLength + pageBorderX
                    originY = (blockY - originY) * squareLength + pageBorderY

                context.set_source_rgba(0.0, 0.0, 0.0, 1.0)
                context.rectangle(originX, originY, markerLength, markerLength)
                context.fill()

                # Generate marker
                if  (mode == "CHARUCO"):
                    markerID = firstMarkerID + (blockY * chessboardSize[0] + blockX) // 2
                elif (mode == "ARUCO"):
                    markerID = firstMarkerID
                elif (mode == "ARUCOGRID"):
                    markerID = firstMarkerID + (blockY * chessboardSize[0] + blockX)

                marker = MarkerPrinter.ArucoBits(dictionary, markerID)
                markerSize = marker.shape[0]
                unitLength = markerLength / (float)(markerSize + borderBits * 2)

                markerBitMap = np.zeros(shape = (markerSize+borderBits*2, markerSize+borderBits*2), dtype = bool)
                markerBitMap[borderBits:-borderBits,borderBits:-borderBits] = marker
                markerBitMap = np.swapaxes(markerBitMap, 0, 1)

                # Compute edges
                hEdges = np.zeros(shape = (markerSize+1,markerSize+1), dtype = bool)
                vEdges = np.zeros(shape = (markerSize+1,markerSize+1), dtype = bool)

                for mx in range(markerSize):
                    for my in range(markerSize+1):
                        if ( markerBitMap[mx + borderBits, my + borderBits - 1] ^ markerBitMap[mx + borderBits, my + borderBits]):
                            hEdges[mx, my] = True

                for mx in range(markerSize+1):
                    for my in range(markerSize):
                        if ( markerBitMap[mx + borderBits - 1, my + borderBits] ^ markerBitMap[mx + borderBits, my + borderBits]):
                            vEdges[mx, my] = True

                # Use for debug, check edge or position is correct or not
                if(MarkerPrinter.debugMode is not None):
                    if(MarkerPrinter.debugMode.upper() == "LINE"):
                        context.set_source_rgba(1.0, 1.0, 1.0, 1.0)
                        context.set_line_width(unitLength * 0.1)
                        for mx in range(markerSize+1):
                            for my in range(markerSize+1):
                                if(hEdges[mx, my]):
                                    context.move_to(originX + unitLength * (mx + borderBits    ), originY + unitLength * (my + borderBits    ))
                                    context.line_to(originX + unitLength * (mx + borderBits + 1), originY + unitLength * (my + borderBits    ))
                                    context.stroke()
                                if(vEdges[mx, my]):
                                    context.move_to(originX + unitLength * (mx + borderBits    ), originY + unitLength * (my + borderBits    ))
                                    context.line_to(originX + unitLength * (mx + borderBits    ), originY + unitLength * (my + borderBits + 1))
                                    context.stroke()

                    elif(MarkerPrinter.debugMode.upper() == "BLOCK"):
                        context.set_source_rgba(1.0, 1.0, 1.0, 1.0)
                        for mx in range(markerSize):
                            for my in range(markerSize):
                                if(markerBitMap[mx + borderBits, my + borderBits]):
                                    context.rectangle(
                                        originX + unitLength * (mx + borderBits),
                                        originY + unitLength * (my + borderBits),
                                        unitLength, unitLength)
                                    context.fill()

                else:
                    while(True):
                        found = False

                        # Find start position
                        sx = 0
                        sy = 0
                        for my in range(markerSize):
                            for mx in range(markerSize):
                                if(hEdges[mx, my]):
                                    found = True
                                    sx = mx
                                    sy = my
                                    if(markerBitMap[sx + borderBits, sy + borderBits - 1]):
                                        context.set_source_rgba(0.0, 0.0, 0.0, 1.0)
                                    else:
                                        context.set_source_rgba(1.0, 1.0, 1.0, 1.0)
                                    break
                            if(found):
                                break

                        context.move_to (originX + unitLength * (sx + borderBits), originY + unitLength * (sy + borderBits))

                        # Use wall follower maze solving algorithm to draw white part
                        cx = sx
                        cy = sy
                        cd = 3 # 0 right, 1 down, 2 left, 3 up
                        while(True):
                            nd = (cd + 1)%4
                            moved = False
                            if(nd == 0):
                                if(hEdges[cx, cy]):
                                    hEdges[cx, cy] = False
                                    cx = cx + 1
                                    moved = True
                            elif(nd == 1):
                                if(vEdges[cx, cy]):
                                    vEdges[cx, cy] = False
                                    cy = cy + 1
                                    moved = True
                            elif(nd == 2):
                                if(hEdges[cx - 1, cy]):
                                    hEdges[cx - 1, cy] = False
                                    cx = cx - 1
                                    moved = True
                            elif(nd == 3):
                                if(vEdges[cx, cy - 1]):
                                    vEdges[cx, cy - 1] = False
                                    cy = cy - 1
                                    moved = True

                            if((cx == sx) and (cy == sy)):
                                context.close_path ()
                                break
                            else:
                                if(moved):
                                    context.line_to(originX + unitLength * (cx + borderBits), originY + unitLength * (cy + borderBits))
                                cd = nd

                        if (found):
                            context.fill()
                        else:
                            break

        else:
            originX = (blockX - originX) * squareLength + pageBorderX
            originY = (blockY - originY) * squareLength + pageBorderY
            context.set_source_rgba(0.0, 0.0, 0.0, 1.0)
            context.rectangle(originX, originY, squareLength, squareLength)
            context.fill()

    def __CheckChessMarkerImage(chessboardSize, squareLength, subSize=None, pageBorder=(0,0)):
        if(len(chessboardSize) != 2):
            raise ValueError("len(chessboardSize) != 2")
        else:
            sizeX, sizeY = chessboardSize

        if(len(pageBorder) != 2):
            raise ValueError("len(pageBorder) != 2")
        else:
            pageBorderX, pageBorderY = pageBorder

        if(sizeX <= 1):
            raise ValueError("sizeX <= 1")

        if(sizeY <= 1):
            raise ValueError("sizeY <= 1")

        if(squareLength <= 0):
            raise ValueError("squareLength <= 0")

        if(pageBorderX < 0):
            raise ValueError("pageBorderX < 0")

        if(pageBorderY < 0):
            raise ValueError("pageBorderY < 0")

        if(subSize is not None):
            subSizeX, subSizeY = subSize

            if(subSizeX < 0):
                raise ValueError("subSizeX < 0")

            if(subSizeY < 0):
                raise ValueError("subSizeY < 0")

    def PreviewChessMarkerImage(chessboardSize, squareLength, pageBorder=(0, 0), dpi=96):
        MarkerPrinter.__CheckChessMarkerImage(chessboardSize, squareLength, pageBorder=pageBorder)

        squareLength = squareLength * MarkerPrinter.ptPerMeter
        pageBorder = (pageBorder[0] * MarkerPrinter.ptPerMeter, pageBorder[1] * MarkerPrinter.ptPerMeter)

        prevImage = None
        with tempfile.TemporaryDirectory() as tmpdirname:
            with MarkerPrinter.surface[".SVG"] (
                os.path.join(tmpdirname, "tempSVG.svg"),
                chessboardSize[0] * squareLength + pageBorder[0] * 2,
                chessboardSize[1] * squareLength + pageBorder[1] * 2) as surface:
                context = cairo.Context(surface)

                context.set_source_rgba(0.5, 0.5, 0.5, 1.0)
                context.rectangle(0, 0,
                    chessboardSize[0] * squareLength + pageBorder[0] * 2,
                    chessboardSize[1] * squareLength + pageBorder[1] * 2)
                context.fill()

                context.set_source_rgba(1.0, 1.0, 1.0, 1.0)
                context.rectangle(pageBorder[0], pageBorder[1],
                    chessboardSize[0] * squareLength,
                    chessboardSize[1] * squareLength)
                context.fill()

                for bx in range(chessboardSize[0]):
                    for by in range(chessboardSize[1]):
                        MarkerPrinter.__DrawBlock(
                            context = context,
                            chessboardSize = chessboardSize,
                            squareLength = squareLength,
                            blockX = bx,
                            blockY = by,
                            pageBorderX = pageBorder[0],
                            pageBorderY = pageBorder[1],
                            mode = "CHESS")

            with open(os.path.join(tmpdirname, "tempSVG.svg")) as file:
                prevImage = Image.open(io.BytesIO(svg2png(bytestring=file.read(), dpi=dpi)))

        return prevImage

    def GenChessMarkerImage(filePath, chessboardSize, squareLength, subSize=None, pageBorder=(0, 0)):
        MarkerPrinter.__CheckChessMarkerImage(chessboardSize, squareLength, subSize=subSize, pageBorder=pageBorder)

        squareLength = squareLength * MarkerPrinter.ptPerMeter
        pageBorder = (pageBorder[0] * MarkerPrinter.ptPerMeter, pageBorder[1] * MarkerPrinter.ptPerMeter)

        # Check
        path, nameExt = os.path.split(filePath)
        name, ext = os.path.splitext(nameExt)

        if(len(path) > 0):
            if not(os.path.isdir(path)):
                os.makedirs(path)

        if((ext.upper() != ".SVG") and (ext.upper() != ".PS") and (ext.upper() != ".PDF")):
            raise ValueError("file extention is not supported, should be: svg, ps, pdf")

        # Draw
        with MarkerPrinter.surface[ext.upper()] (
            filePath,
            chessboardSize[0] * squareLength + pageBorder[0] * 2,
            chessboardSize[1] * squareLength + pageBorder[1] * 2) as surface:
            context = cairo.Context(surface)

            context.set_source_rgba(0.5, 0.5, 0.5, 1.0)
            context.rectangle(0, 0,
                chessboardSize[0] * squareLength + pageBorder[0] * 2,
                chessboardSize[1] * squareLength + pageBorder[1] * 2)
            context.fill()

            context.set_source_rgba(1.0, 1.0, 1.0, 1.0)
            context.rectangle(pageBorder[0], pageBorder[1],
                chessboardSize[0] * squareLength,
                chessboardSize[1] * squareLength)
            context.fill()

            for bx in range(chessboardSize[0]):
                for by in range(chessboardSize[1]):
                    MarkerPrinter.__DrawBlock(
                        context = context,
                        chessboardSize = chessboardSize,
                        squareLength = squareLength,
                        blockX = bx,
                        blockY = by,
                        pageBorderX = pageBorder[0],
                        pageBorderY = pageBorder[1],
                        mode = "CHESS" )

        if(subSize is not None):
            subDivide = (\
                chessboardSize[0] // subSize[0] + int(chessboardSize[0] % subSize[0] > 0),
                chessboardSize[1] // subSize[1] + int(chessboardSize[1] % subSize[1] > 0))

            subChessboardBlockX = np.clip ( np.arange(0, subSize[0] * subDivide[0] + 1, subSize[0]), 0, chessboardSize[0])
            subChessboardBlockY = np.clip ( np.arange(0, subSize[1] * subDivide[1] + 1, subSize[1]), 0, chessboardSize[1])

            subChessboardSliceX = subChessboardBlockX.astype(float) * squareLength
            subChessboardSliceY = subChessboardBlockY.astype(float) * squareLength

            for subXID in range(subDivide[0]):
                for subYID in range(subDivide[1]):
                    subName = name + \
                        "_X" + str(subChessboardBlockX[subXID]) + "_" + str(subChessboardBlockX[subXID+1]) + \
                        "_Y" + str(subChessboardBlockY[subYID]) + "_" + str(subChessboardBlockY[subYID+1])

                    with MarkerPrinter.surface[ext.upper()](
                        os.path.join(path, subName + ext),
                        subChessboardSliceX[subXID+1] - subChessboardSliceX[subXID] + pageBorder[0] * 2,
                        subChessboardSliceY[subYID+1] - subChessboardSliceY[subYID] + pageBorder[1] * 2) as surface:
                        context = cairo.Context(surface)

                        context.set_source_rgba(0.5, 0.5, 0.5, 1.0)
                        context.rectangle(0, 0,
                            subChessboardSliceX[subXID+1] - subChessboardSliceX[subXID] + pageBorder[0] * 2,
                            subChessboardSliceY[subYID+1] - subChessboardSliceY[subYID] + pageBorder[1] * 2)
                        context.fill()

                        context.set_source_rgba(1.0, 1.0, 1.0, 1.0)
                        context.rectangle(pageBorder[0], pageBorder[1],
                            subChessboardSliceX[subXID+1] - subChessboardSliceX[subXID],
                            subChessboardSliceY[subYID+1] - subChessboardSliceY[subYID])
                        context.fill()

                        for bx in range(subChessboardBlockX[subXID+1] - subChessboardBlockX[subXID]):
                            for by in range(subChessboardBlockY[subYID+1] - subChessboardBlockY[subYID]):
                                MarkerPrinter.__DrawBlock(
                                    context = context,
                                    chessboardSize = chessboardSize,
                                    squareLength = squareLength,
                                    blockX = subChessboardBlockX[subXID] + bx,
                                    blockY = subChessboardBlockY[subYID] + by,
                                    originX = subChessboardBlockX[subXID],
                                    originY = subChessboardBlockY[subYID],
                                    pageBorderX = pageBorder[0],
                                    pageBorderY = pageBorder[1],
                                    mode = "CHESS" )


    def __CheckArucoMarkerImage(dictionary, markerID, markerLength, borderBits=1, pageBorder=(0, 0)):
        if(len(pageBorder) != 2):
            raise ValueError("len(pageBorder) != 2")
        else:
            pageBorderX, pageBorderY = pageBorder

        if not (dictionary in MarkerPrinter.arucoDictBytesList):
            raise ValueError("dictionary is not support")

        if(MarkerPrinter.arucoDictBytesList[dictionary].shape[0] <= markerID ):
            raise ValueError("markerID is not in aruce dictionary")

        if(markerID < 0):
            raise ValueError("markerID < 0")

        if(markerLength <= 0):
            raise ValueError("markerLength <= 0")

        if(borderBits <= 0):
            raise ValueError("borderBits <= 0")

        if(pageBorderX < 0):
            raise ValueError("pageBorderX < 0")

        if(pageBorderY < 0):
            raise ValueError("pageBorderY < 0")

    def PreviewArucoMarkerImage(dictionary, markerID, markerLength, borderBits=1, pageBorder=(0, 0), dpi=96):
        MarkerPrinter.__CheckArucoMarkerImage(dictionary, markerID, markerLength, borderBits=borderBits, pageBorder=pageBorder)

        markerLength = markerLength * MarkerPrinter.ptPerMeter
        pageBorder = (pageBorder[0] * MarkerPrinter.ptPerMeter, pageBorder[1] * MarkerPrinter.ptPerMeter)

        prevImage = None
        with tempfile.TemporaryDirectory() as tmpdirname:
            with MarkerPrinter.surface[".SVG"] (
                os.path.join(tmpdirname, "tempSVG.svg"),
                markerLength + pageBorder[0] * 2,
                markerLength + pageBorder[1] * 2) as surface:
                context = cairo.Context(surface)

                context.set_source_rgba(0.5, 0.5, 0.5, 1.0)
                context.rectangle(0, 0,
                    markerLength + pageBorder[0] * 2,
                    markerLength + pageBorder[1] * 2)
                context.fill()

                context.set_source_rgba(1.0, 1.0, 1.0, 1.0)
                context.rectangle(pageBorder[0], pageBorder[1],
                    markerLength,
                    markerLength)
                context.fill()

                MarkerPrinter.__DrawBlock(
                    context = context,
                    dictionary = dictionary,
                    markerLength = markerLength,
                    borderBits = borderBits,
                    firstMarkerID = markerID,
                    pageBorderX = pageBorder[0],
                    pageBorderY = pageBorder[1],
                    mode = "ARUCO")

            with open(os.path.join(tmpdirname, "tempSVG.svg")) as file:
                prevImage = Image.open(io.BytesIO(svg2png(bytestring=file.read(), dpi=dpi)))

        return prevImage

    def GenArucoMarkerImage(filePath, dictionary, markerID, markerLength, borderBits=1, pageBorder=(0, 0)):
        MarkerPrinter.__CheckArucoMarkerImage(dictionary, markerID, markerLength, borderBits=borderBits, pageBorder=pageBorder)

        markerLength = markerLength * MarkerPrinter.ptPerMeter
        pageBorder = (pageBorder[0] * MarkerPrinter.ptPerMeter, pageBorder[1] * MarkerPrinter.ptPerMeter)

        # Check
        path, nameExt = os.path.split(filePath)
        name, ext = os.path.splitext(nameExt)

        if(len(path) > 0):
            if not(os.path.isdir(path)):
                os.makedirs(path)

        if((ext.upper() != ".SVG") and (ext.upper() != ".PS") and (ext.upper() != ".PDF")):
            raise ValueError("file extention is not supported, should be: svg, ps, pdf")

        # Draw
        with MarkerPrinter.surface[ext.upper()] (
            filePath,
            markerLength + pageBorder[0] * 2,
            markerLength + pageBorder[1] * 2) as surface:
            context = cairo.Context(surface)

            context.set_source_rgba(0.5, 0.5, 0.5, 1.0)
            context.rectangle(0, 0,
                markerLength + pageBorder[0] * 2,
                markerLength + pageBorder[1] * 2)
            context.fill()

            context.set_source_rgba(1.0, 1.0, 1.0, 1.0)
            context.rectangle(pageBorder[0], pageBorder[1],
                markerLength,
                markerLength)
            context.fill()

            MarkerPrinter.__DrawBlock(
                context = context,
                dictionary = dictionary,
                markerLength = markerLength,
                borderBits = borderBits,
                firstMarkerID = markerID,
                pageBorderX = pageBorder[0],
                pageBorderY = pageBorder[1],
                mode = "ARUCO")

    def __CheckCharucoMarkerImage(dictionary, chessboardSize, squareLength, markerLength, borderBits=1, subSize=None, pageBorder=(0, 0)):
        if(len(chessboardSize) != 2):
            raise ValueError("len(chessboardSize) != 2")
        else:
            sizeX, sizeY = chessboardSize

        if(len(pageBorder) != 2):
            raise ValueError("len(pageBorder) != 2")
        else:
            pageBorderX, pageBorderY = pageBorder

        if not (dictionary in MarkerPrinter.arucoDictBytesList):
            raise ValueError("dictionary is not support")

        if(MarkerPrinter.arucoDictBytesList[dictionary].shape[0] < (( sizeX * sizeY ) // 2)):
            raise ValueError("aruce dictionary is not enough for your board size")

        if(sizeX <= 1):
            raise ValueError("sizeX <= 1")

        if(sizeY <= 1):
            raise ValueError("sizeY <= 1")

        if(squareLength <= 0):
            raise ValueError("squareLength <= 0")

        if(markerLength <= 0):
            raise ValueError("markerLength <= 0")

        if(squareLength < markerLength):
            raise ValueError("squareLength < markerLength")

        if(borderBits <= 0):
            raise ValueError("borderBits <= 0")

        if(pageBorderX < 0):
            raise ValueError("pageBorderX < 0")

        if(pageBorderY < 0):
            raise ValueError("pageBorderY < 0")

        if(subSize is not None):
            subSizeX, subSizeY = subSize

            if(subSizeX < 0):
                raise ValueError("subSizeX < 0")

            if(subSizeY < 0):
                raise ValueError("subSizeY < 0")

    def PreviewCharucoMarkerImage(dictionary, chessboardSize, squareLength, markerLength, borderBits=1, pageBorder=(0, 0), dpi=96):
        MarkerPrinter.__CheckCharucoMarkerImage(dictionary, chessboardSize, squareLength, markerLength, borderBits=borderBits, pageBorder=pageBorder)

        squareLength = squareLength * MarkerPrinter.ptPerMeter
        markerLength = markerLength * MarkerPrinter.ptPerMeter
        pageBorder = (pageBorder[0] * MarkerPrinter.ptPerMeter, pageBorder[1] * MarkerPrinter.ptPerMeter)

        prevImage = None
        with tempfile.TemporaryDirectory() as tmpdirname:
            with MarkerPrinter.surface[".SVG"] (
                os.path.join(tmpdirname, "tempSVG.svg"),
                chessboardSize[0] * squareLength + pageBorder[0] * 2,
                chessboardSize[1] * squareLength + pageBorder[1] * 2) as surface:
                context = cairo.Context(surface)

                context.set_source_rgba(0.5, 0.5, 0.5, 1.0)
                context.rectangle(0, 0,
                    chessboardSize[0] * squareLength + pageBorder[0] * 2,
                    chessboardSize[1] * squareLength + pageBorder[1] * 2)
                context.fill()

                context.set_source_rgba(1.0, 1.0, 1.0, 1.0)
                context.rectangle(pageBorder[0], pageBorder[1],
                    chessboardSize[0] * squareLength,
                    chessboardSize[1] * squareLength)
                context.fill()

                for bx in range(chessboardSize[0]):
                    for by in range(chessboardSize[1]):
                        MarkerPrinter.__DrawBlock(
                            context = context,
                            dictionary = dictionary,
                            markerLength = markerLength,
                            borderBits = borderBits,
                            chessboardSize = chessboardSize,
                            squareLength = squareLength,
                            blockX = bx,
                            blockY = by,
                            pageBorderX = pageBorder[0],
                            pageBorderY = pageBorder[1],
                            mode = "CHARUCO")

            with open(os.path.join(tmpdirname, "tempSVG.svg")) as file:
                prevImage = Image.open(io.BytesIO(svg2png(bytestring=file.read(), dpi=dpi)))

        return prevImage

    def GenCharucoMarkerImage(filePath, dictionary, chessboardSize, squareLength, markerLength, borderBits=1, subSize=None, pageBorder=(0, 0)):
        MarkerPrinter.__CheckCharucoMarkerImage(dictionary, chessboardSize, squareLength, markerLength, borderBits=borderBits, subSize=subSize, pageBorder=pageBorder)

        squareLength = squareLength * MarkerPrinter.ptPerMeter
        markerLength = markerLength * MarkerPrinter.ptPerMeter
        pageBorder = (pageBorder[0] * MarkerPrinter.ptPerMeter, pageBorder[1] * MarkerPrinter.ptPerMeter)

        # Check
        path, nameExt = os.path.split(filePath)
        name, ext = os.path.splitext(nameExt)

        if(len(path) > 0):
            if not(os.path.isdir(path)):
                os.makedirs(path)

        if((ext.upper() != ".SVG") and (ext.upper() != ".PS") and (ext.upper() != ".PDF")):
            raise ValueError("file extention is not supported, should be: svg, ps, pdf")

        # Draw
        with MarkerPrinter.surface[ext.upper()] (
            filePath,
            chessboardSize[0] * squareLength + pageBorder[0] * 2,
            chessboardSize[1] * squareLength + pageBorder[1] * 2) as surface:
            context = cairo.Context(surface)

            context.set_source_rgba(0.5, 0.5, 0.5, 1.0)
            context.rectangle(0, 0,
                chessboardSize[0] * squareLength + pageBorder[0] * 2,
                chessboardSize[1] * squareLength + pageBorder[1] * 2)
            context.fill()

            context.set_source_rgba(1.0, 1.0, 1.0, 1.0)
            context.rectangle(pageBorder[0], pageBorder[1],
                chessboardSize[0] * squareLength,
                chessboardSize[1] * squareLength)
            context.fill()

            for bx in range(chessboardSize[0]):
                for by in range(chessboardSize[1]):
                    MarkerPrinter.__DrawBlock(
                        context = context,
                        dictionary = dictionary,
                        markerLength = markerLength,
                        borderBits = borderBits,
                        chessboardSize = chessboardSize,
                        squareLength = squareLength,
                        blockX = bx,
                        blockY = by,
                        pageBorderX = pageBorder[0],
                        pageBorderY = pageBorder[1],
                        mode = "CHARUCO")

        if(subSize is not None):
            subDivide = (\
                chessboardSize[0] // subSize[0] + int(chessboardSize[0] % subSize[0] > 0),
                chessboardSize[1] // subSize[1] + int(chessboardSize[1] % subSize[1] > 0))

            subChessboardBlockX = np.clip ( np.arange(0, subSize[0] * subDivide[0] + 1, subSize[0]), 0, chessboardSize[0])
            subChessboardBlockY = np.clip ( np.arange(0, subSize[1] * subDivide[1] + 1, subSize[1]), 0, chessboardSize[1])

            subChessboardSliceX = subChessboardBlockX.astype(float) * squareLength
            subChessboardSliceY = subChessboardBlockY.astype(float) * squareLength

            for subXID in range(subDivide[0]):
                for subYID in range(subDivide[1]):
                    subName = name + \
                        "_X" + str(subChessboardBlockX[subXID]) + "_" + str(subChessboardBlockX[subXID+1]) + \
                        "_Y" + str(subChessboardBlockY[subYID]) + "_" + str(subChessboardBlockY[subYID+1])

                    with MarkerPrinter.surface[ext.upper()](
                        os.path.join(path, subName + ext),
                        subChessboardSliceX[subXID+1] - subChessboardSliceX[subXID] + pageBorder[0] * 2,
                        subChessboardSliceY[subYID+1] - subChessboardSliceY[subYID] + pageBorder[1] * 2) as surface:
                        context = cairo.Context(surface)

                        context.set_source_rgba(0.5, 0.5, 0.5, 1.0)
                        context.rectangle(0, 0,
                            subChessboardSliceX[subXID+1] - subChessboardSliceX[subXID] + pageBorder[0] * 2,
                            subChessboardSliceY[subYID+1] - subChessboardSliceY[subYID] + pageBorder[1] * 2)
                        context.fill()

                        context.set_source_rgba(1.0, 1.0, 1.0, 1.0)
                        context.rectangle(pageBorder[0], pageBorder[1],
                            subChessboardSliceX[subXID+1] - subChessboardSliceX[subXID],
                            subChessboardSliceY[subYID+1] - subChessboardSliceY[subYID])
                        context.fill()

                        for bx in range(subChessboardBlockX[subXID+1] - subChessboardBlockX[subXID]):
                            for by in range(subChessboardBlockY[subYID+1] - subChessboardBlockY[subYID]):
                                MarkerPrinter.__DrawBlock(
                                    context = context,
                                    dictionary = dictionary,
                                    markerLength = markerLength,
                                    borderBits = borderBits,
                                    chessboardSize = chessboardSize,
                                    squareLength = squareLength,
                                    blockX = subChessboardBlockX[subXID] + bx,
                                    blockY = subChessboardBlockY[subYID] + by,
                                    originX = subChessboardBlockX[subXID],
                                    originY = subChessboardBlockY[subYID],
                                    pageBorderX = pageBorder[0],
                                    pageBorderY = pageBorder[1],
                                    mode = "CHARUCO")

    def __CheckArucoGridMarkerImage(dictionary, chessboardSize, markerLength, markerSeparation, firstMarker, borderBits=1, subSize=None, pageBorder=(0, 0)):
        if(len(chessboardSize) != 2):
            raise ValueError("len(chessboardSize) != 2")
        else:
            sizeX, sizeY = chessboardSize

        if(len(pageBorder) != 2):
            raise ValueError("len(pageBorder) != 2")
        else:
            pageBorderX, pageBorderY = pageBorder

        if not (dictionary in MarkerPrinter.arucoDictBytesList):
            raise ValueError("dictionary is not support")

        if(MarkerPrinter.arucoDictBytesList[dictionary].shape[0] < (( sizeX * sizeY ) + firstMarker)):
            raise ValueError("aruce dictionary is not enough for your board size and firstMarker")

        if(sizeX <= 1):
            raise ValueError("sizeX <= 1")

        if(sizeY <= 1):
            raise ValueError("sizeY <= 1")

        if(markerLength <= 0):
            raise ValueError("markerLength <= 0")

        if(markerSeparation <= 0):
            raise ValueError("markerSeparation <= 0")

        if(borderBits <= 0):
            raise ValueError("borderBits <= 0")

        if(pageBorderX < 0):
            raise ValueError("pageBorderX < 0")

        if(pageBorderY < 0):
            raise ValueError("pageBorderY < 0")

        if(subSize is not None):
            subSizeX, subSizeY = subSize

            if(subSizeX < 0):
                raise ValueError("subSizeX < 0")

            if(subSizeY < 0):
                raise ValueError("subSizeY < 0")

    def PreviewArucoGridMarkerImage(dictionary, chessboardSize, markerLength, markerSeparation, firstMarker, borderBits=1, pageBorder=(0, 0), dpi=96):
        MarkerPrinter.__CheckArucoGridMarkerImage(dictionary, chessboardSize, markerLength, markerSeparation, firstMarker, borderBits=borderBits, pageBorder=pageBorder)

        markerLength = markerLength * MarkerPrinter.ptPerMeter
        markerSeparation = markerSeparation * MarkerPrinter.ptPerMeter
        pageBorder = (pageBorder[0] * MarkerPrinter.ptPerMeter, pageBorder[1] * MarkerPrinter.ptPerMeter)

        prevImage = None
        with tempfile.TemporaryDirectory() as tmpdirname:
            with MarkerPrinter.surface[".SVG"] (
                os.path.join(tmpdirname, "tempSVG.svg"),
                chessboardSize[0] * markerLength + (chessboardSize[0] - 1) * markerSeparation + pageBorder[0] * 2,
                chessboardSize[1] * markerLength + (chessboardSize[1] - 1) * markerSeparation + pageBorder[1] * 2) as surface:
                context = cairo.Context(surface)

                context.set_source_rgba(0.5, 0.5, 0.5, 1.0)
                context.rectangle(0, 0,
                    chessboardSize[0] * markerLength + (chessboardSize[0] - 1) * markerSeparation + pageBorder[0] * 2,
                    chessboardSize[1] * markerLength + (chessboardSize[1] - 1) * markerSeparation + pageBorder[1] * 2)
                context.fill()

                context.set_source_rgba(1.0, 1.0, 1.0, 1.0)
                context.rectangle(pageBorder[0], pageBorder[1],
                    chessboardSize[0] * markerLength + (chessboardSize[0] - 1) * markerSeparation,
                    chessboardSize[1] * markerLength + (chessboardSize[1] - 1) * markerSeparation)
                context.fill()

                for bx in range(chessboardSize[0]):
                    for by in range(chessboardSize[1]):
                        MarkerPrinter.__DrawBlock(
                            context = context,
                            dictionary = dictionary,
                            markerLength = markerLength,
                            borderBits = borderBits,
                            chessboardSize = chessboardSize,
                            squareLength = markerLength + markerSeparation,
                            firstMarkerID = firstMarker,
                            blockX = bx,
                            blockY = by,
                            pageBorderX = pageBorder[0],
                            pageBorderY = pageBorder[1],
                            mode = "ARUCOGRID")

            with open(os.path.join(tmpdirname, "tempSVG.svg")) as file:
                prevImage = Image.open(io.BytesIO(svg2png(bytestring=file.read(), dpi=dpi)))

        return prevImage

    def GenArucoGridMarkerImage(filePath, dictionary, chessboardSize, markerLength, markerSeparation, firstMarker, borderBits=1, subSize=None, pageBorder=(0, 0)):
        MarkerPrinter.__CheckArucoGridMarkerImage(dictionary, chessboardSize, markerLength, markerSeparation, firstMarker, borderBits=borderBits, subSize=subSize, pageBorder=pageBorder)

        markerLength = markerLength * MarkerPrinter.ptPerMeter
        markerSeparation = markerSeparation * MarkerPrinter.ptPerMeter
        pageBorder = (pageBorder[0] * MarkerPrinter.ptPerMeter, pageBorder[1] * MarkerPrinter.ptPerMeter)

        # Check
        path, nameExt = os.path.split(filePath)
        name, ext = os.path.splitext(nameExt)

        if(len(path) > 0):
            if not(os.path.isdir(path)):
                os.makedirs(path)

        if((ext.upper() != ".SVG") and (ext.upper() != ".PS") and (ext.upper() != ".PDF")):
            raise ValueError("file extention is not supported, should be: svg, ps, pdf")

        # Draw
        with MarkerPrinter.surface[ext.upper()] (
            filePath,
            chessboardSize[0] * markerLength + (chessboardSize[0] - 1) * markerSeparation + pageBorder[0] * 2,
            chessboardSize[1] * markerLength + (chessboardSize[1] - 1) * markerSeparation + pageBorder[1] * 2) as surface:
            context = cairo.Context(surface)

            context.set_source_rgba(0.5, 0.5, 0.5, 1.0)
            context.rectangle(0, 0,
                chessboardSize[0] * markerLength + (chessboardSize[0] - 1) * markerSeparation + pageBorder[0] * 2,
                chessboardSize[1] * markerLength + (chessboardSize[1] - 1) * markerSeparation + pageBorder[1] * 2)
            context.fill()

            context.set_source_rgba(1.0, 1.0, 1.0, 1.0)
            context.rectangle(pageBorder[0], pageBorder[1],
                chessboardSize[0] * markerLength + (chessboardSize[0] - 1) * markerSeparation,
                chessboardSize[1] * markerLength + (chessboardSize[1] - 1) * markerSeparation)
            context.fill()

            for bx in range(chessboardSize[0]):
                for by in range(chessboardSize[1]):
                    MarkerPrinter.__DrawBlock(
                        context = context,
                        dictionary = dictionary,
                        markerLength = markerLength,
                        borderBits = borderBits,
                        chessboardSize = chessboardSize,
                        squareLength = markerLength + markerSeparation,
                        firstMarkerID = firstMarker,
                        blockX = bx,
                        blockY = by,
                        pageBorderX = pageBorder[0],
                        pageBorderY = pageBorder[1],
                        mode = "ARUCOGRID")

        if(subSize is not None):
            subDivide = (\
                chessboardSize[0] // subSize[0] + int(chessboardSize[0] % subSize[0] > 0),
                chessboardSize[1] // subSize[1] + int(chessboardSize[1] % subSize[1] > 0))

            subChessboardBlockX = np.clip ( np.arange(0, subSize[0] * subDivide[0] + 1, subSize[0]), 0, chessboardSize[0])
            subChessboardBlockY = np.clip ( np.arange(0, subSize[1] * subDivide[1] + 1, subSize[1]), 0, chessboardSize[1])

            subChessboardSliceX = subChessboardBlockX.astype(float) * (markerLength + markerSeparation)
            subChessboardSliceY = subChessboardBlockY.astype(float) * (markerLength + markerSeparation)

            subChessboardSliceX[-1] -= markerSeparation
            subChessboardSliceY[-1] -= markerSeparation

            for subXID in range(subDivide[0]):
                for subYID in range(subDivide[1]):
                    subName = name + \
                        "_X" + str(subChessboardBlockX[subXID]) + "_" + str(subChessboardBlockX[subXID+1]) + \
                        "_Y" + str(subChessboardBlockY[subYID]) + "_" + str(subChessboardBlockY[subYID+1])

                    with MarkerPrinter.surface[ext.upper()](
                        os.path.join(path, subName + ext),
                        subChessboardSliceX[subXID+1] - subChessboardSliceX[subXID] + pageBorder[0] * 2,
                        subChessboardSliceY[subYID+1] - subChessboardSliceY[subYID] + pageBorder[1] * 2) as surface:
                        context = cairo.Context(surface)

                        context.set_source_rgba(0.5, 0.5, 0.5, 1.0)
                        context.rectangle(0, 0,
                            subChessboardSliceX[subXID+1] - subChessboardSliceX[subXID] + pageBorder[0] * 2,
                            subChessboardSliceY[subYID+1] - subChessboardSliceY[subYID] + pageBorder[1] * 2)
                        context.fill()

                        context.set_source_rgba(1.0, 1.0, 1.0, 1.0)
                        context.rectangle(pageBorder[0], pageBorder[1],
                            subChessboardSliceX[subXID+1] - subChessboardSliceX[subXID],
                            subChessboardSliceY[subYID+1] - subChessboardSliceY[subYID])
                        context.fill()

                        for bx in range(subChessboardBlockX[subXID+1] - subChessboardBlockX[subXID]):
                            for by in range(subChessboardBlockY[subYID+1] - subChessboardBlockY[subYID]):
                                MarkerPrinter.__DrawBlock(
                                    context = context,
                                    dictionary = dictionary,
                                    markerLength = markerLength,
                                    borderBits = borderBits,
                                    chessboardSize = chessboardSize,
                                    squareLength = markerLength + markerSeparation,
                                    firstMarkerID = firstMarker,
                                    blockX = subChessboardBlockX[subXID] + bx,
                                    blockY = subChessboardBlockY[subYID] + by,
                                    originX = subChessboardBlockX[subXID],
                                    originY = subChessboardBlockY[subYID],
                                    pageBorderX = pageBorder[0],
                                    pageBorderY = pageBorder[1],
                                    mode = "ARUCOGRID")

if __name__ == '__main__':
    parser = ArgumentParser()

    # Save marker image parameters
    chessGroup = parser.add_argument_group('chess', 'Chessboard')
    arucoGroup = parser.add_argument_group('aruco', 'ArUco')
    arucoGridGroup = parser.add_argument_group('aruco_grid', 'ArUco grid')
    charucoGroup = parser.add_argument_group('charuco', 'ChArUco')
    exclusiveGroup = parser.add_mutually_exclusive_group()

    exclusiveGroup.add_argument(
        "--chess", action='store_true', default=False,
        help="Choose to save chessboard marker")

    exclusiveGroup.add_argument(
        "--aruco", action='store_true', default=False,
        help="Choose to save ArUco marker")

    exclusiveGroup.add_argument(
        "--aruco_grid", action='store_true', default=False,
        help="Choose to save ArUco grid marker")

    exclusiveGroup.add_argument(
        "--charuco", action='store_true', default=False,
        help="Choose to save ChArUco marker")

    # Utility functions parameters
    exclusiveGroup.add_argument(
        "--generate", dest="arucoDataFileName",
        help="Generate aruco data to FILE", metavar="FILE")

    exclusiveGroup.add_argument(
        "--list_dictionary", action='store_true', default=False,
        help="List predefined aruco dictionary")

    # Parameters
    # fileName
    parser.add_argument(
        "--file", dest="fileName", default="./image.pdf",
        help="Save marker image to FILE", metavar="FILE")
    for group in [chessGroup, arucoGroup, arucoGridGroup, charucoGroup]:
        group.add_argument(
            "--" + group.title + "_file", dest="fileName",
            help="Save marker image to FILE", metavar="FILE")

    # dictionary
    parser.add_argument(
        "--dictionary", dest="dictionary", default="DICT_ARUCO_ORIGINAL",
        help="Generate marker via predefined DICTIONARY aruco dictionary", metavar="DICTIONARY")
    for group in [arucoGroup, arucoGridGroup, charucoGroup]:
        group.add_argument(
            "--" + group.title + "_dictionary", dest="dictionary",
            help="Generate marker via predefined DICTIONARY aruco dictionary", metavar="DICTIONARY")

    # size
    parser.add_argument(
        "--size_x", dest="sizeX", default="16",
        help="Save marker image with N board width", metavar="N")
    parser.add_argument(
        "--size_y", dest="sizeY", default="9",
        help="Save marker image with N board height", metavar="N")

    for group in [chessGroup, arucoGridGroup, charucoGroup]:
        group.add_argument(
            "--" + group.title + "_size_x", dest="sizeX",
            help="Save marker image with N board width", metavar="N")
        group.add_argument(
            "--" + group.title + "_size_y", dest="sizeY",
            help="Save marker image with N board height", metavar="N")

    # length
    parser.add_argument(
        "--square_length", dest="squareLength", default="0.09",
        help="Save marker image with L square length (Unit: meter)", metavar="L")
    parser.add_argument(
        "--marker_length", dest="markerLength", default="0.07",
        help="Save marker image with L marker length (Unit: meter)", metavar="L")
    parser.add_argument(
        "--marker_separation", dest="markerSeparation", default="0.02",
        help="Save marker image with L separation length (Unit: meter)", metavar="L")

    for group in [chessGroup, charucoGroup]:
        group.add_argument(
            "--" + group.title + "_square_length", dest="squareLength",
            help="Save marker image with L blocks length (Unit: meter)", metavar="L")

    for group in [arucoGroup, arucoGridGroup, charucoGroup]:
        group.add_argument(
            "--" + group.title + "_marker_length", dest="markerLength",
            help="Save marker image with L marker length (Unit: meter)", metavar="L")

    for group in [arucoGridGroup]:
        group.add_argument(
            "--" + group.title + "_marker_separation", dest="markerSeparation",
            help="Save marker image with L gap length (Unit: meter)", metavar="L")

    # else
    parser.add_argument(
        "--marker_id", dest="markerID", default="0",
        help="Save marker image with ID marker", metavar="ID")
    parser.add_argument(
        "--first_marker", dest="firstMarker", default="0",
        help="Save marker image that start with ID marker", metavar="ID")
    parser.add_argument(
        "--border_bits", dest="borderBits", default="1",
        help="Save marker image with N border size", metavar="N")

    for group in [arucoGroup]:
        group.add_argument(
            "--" + group.title + "_marker_id", dest="markerID",
            help="Save marker image with ID marker", metavar="ID")

    for group in [arucoGridGroup]:
        group.add_argument(
            "--" + group.title + "_first_marker", dest="firstMarker",
            help="Save marker image that start with ID marker", metavar="ID")

    for group in [arucoGroup, arucoGridGroup, charucoGroup]:
        group.add_argument(
            "--" + group.title + "_border_bits", dest="borderBits",
            help="Save marker image with N border size", metavar="N")

    # sub size
    parser.add_argument(
        "--sub_size_x", dest="subSizeX", default="0",
        help="Save marker image with N chuck width", metavar="N")
    parser.add_argument(
        "--sub_size_y", dest="subSizeY", default="0",
        help="Save marker image with N chuck height", metavar="N")

    for group in [chessGroup, arucoGridGroup, charucoGroup]:
        group.add_argument(
            "--" + group.title + "_sub_size_x", dest="subSizeX",
            help="Save marker image with N chuck width", metavar="N")
        group.add_argument(
            "--" + group.title + "_sub_size_y", dest="subSizeY",
            help="Save marker image with N chuck height", metavar="N")

    # page border
    parser.add_argument(
        "--page_border_x", dest="pageBorderX", default="0",
        help="Save with page border width L length (Unit: meter)", metavar="L")
    parser.add_argument(
        "--page_border_y", dest="pageBorderY", default="0",
        help="Save with page border height L length (Unit: meter)", metavar="L")

    for group in [chessGroup, arucoGroup, arucoGridGroup, charucoGroup]:
        group.add_argument(
            "--" + group.title + "_page_border_x", dest="pageBorderX", default="0",
            help="Save with page border width L length (Unit: meter)", metavar="L")
        group.add_argument(
            "--" + group.title + "_page_border_y", dest="pageBorderY", default="0",
            help="Save with page border height L length (Unit: meter)", metavar="L")

    # Run
    args = parser.parse_args()

    if(args.arucoDataFileName is not None):
        print("Generate aruco data to: " + args.arucoDataFileName)
        SaveArucoDictBytesList(args.arucoDataFileName)

    elif(args.list_dictionary):
        print("List predefined aruco dictionary")
        for i in MarkerPrinter.arucoDictBytesList.keys():
            print(i)

    elif(args.chess):
        try:
            sizeX = int(args.sizeX)
            sizeY = int(args.sizeY)
            squareLength = float(args.squareLength)
            subSizeX = int(args.subSizeX)
            subSizeY = int(args.subSizeY)
            pageBorderX = float(args.pageBorderX)
            pageBorderY = float(args.pageBorderY)
        except ValueError as e:
            warnings.warn(str(e))
        else:
            print("Save chessboard marker with parms: " + \
                    str({ \
                        "fileName": args.fileName, \
                        "sizeX": sizeX, \
                        "sizeY": sizeY, \
                        "squareLength": squareLength, \
                        "subSizeX": subSizeX, \
                        "subSizeY": subSizeY, \
                        "pageBorderX": pageBorderX, \
                        "pageBorderY": pageBorderY, \
                    }))

            subSize = None

            if(subSizeX > 0):
                if(subSizeY > 0):
                    subSize = (subSizeX, subSizeY)
                else:
                    subSize = (subSizeX, sizeY)
            else:
                if(subSizeY > 0):
                    subSize = (sizeX, subSizeY)
                else:
                    subSize = None

            # Gen
            MarkerPrinter.GenChessMarkerImage(args.fileName, (sizeX, sizeY), squareLength, subSize = subSize, pageBorder = (pageBorderX, pageBorderY))

    elif(args.aruco):
        try:
            markerLength = float(args.markerLength)
            markerID = int(args.markerID)
            borderBits = int(args.borderBits)
            pageBorderX = float(args.pageBorderX)
            pageBorderY = float(args.pageBorderY)
        except ValueError as e:
            warnings.warn(str(e))
        else:
            print("Save ArUco marker with parms: " + \
                    str({ \
                        "fileName": args.fileName, \
                        "dictionary": args.dictionary, \
                        "markerLength": markerLength, \
                        "markerID": markerID, \
                        "borderBits": borderBits, \
                        "pageBorderX": pageBorderX, \
                        "pageBorderY": pageBorderY, \
                    }))

            # Gen
            MarkerPrinter.GenArucoMarkerImage(args.fileName, args.dictionary, markerID, markerLength, borderBits=borderBits, pageBorder = (pageBorderX, pageBorderY))

    elif(args.aruco_grid):
        try:
            sizeX = int(args.sizeX)
            sizeY = int(args.sizeY)
            markerLength = float(args.markerLength)
            markerSeparation = float(args.markerSeparation)
            firstMarker = int(args.firstMarker)
            borderBits = int(args.borderBits)
            subSizeX = int(args.subSizeX)
            subSizeY = int(args.subSizeY)
            pageBorderX = float(args.pageBorderX)
            pageBorderY = float(args.pageBorderY)
        except ValueError as e:
            warnings.warn(str(e))
        else:
            print("Save ArUco grid marker with parms: " + \
                    str({ \
                        "fileName": args.fileName, \
                        "dictionary": args.dictionary, \
                        "sizeX": sizeX, \
                        "sizeY": sizeY, \
                        "markerLength": markerLength, \
                        "markerSeparation": markerSeparation, \
                        "firstMarker": firstMarker, \
                        "borderBits": borderBits, \
                        "subSizeX": subSizeX, \
                        "subSizeY": subSizeY, \
                        "pageBorderX": pageBorderX, \
                        "pageBorderY": pageBorderY, \
                    }))

            subSize = None

            if(subSizeX > 0):
                if(subSizeY > 0):
                    subSize = (subSizeX, subSizeY)
                else:
                    subSize = (subSizeX, sizeY)
            else:
                if(subSizeY > 0):
                    subSize = (sizeX, subSizeY)
                else:
                    subSize = None

            # Gen
            MarkerPrinter.GenArucoGridMarkerImage(args.fileName, args.dictionary, (sizeX, sizeY), markerLength, markerSeparation, firstMarker, borderBits=borderBits, subSize=subSize, pageBorder = (pageBorderX, pageBorderY))

    elif(args.charuco):
        try:
            sizeX = int(args.sizeX)
            sizeY = int(args.sizeY)
            squareLength = float(args.squareLength)
            markerLength = float(args.markerLength)
            borderBits = int(args.borderBits)
            subSizeX = int(args.subSizeX)
            subSizeY = int(args.subSizeY)
            pageBorderX = float(args.pageBorderX)
            pageBorderY = float(args.pageBorderY)
        except ValueError as e:
            warnings.warn(str(e))
        else:
            print("Save ChArUco marker with parms: " + \
                    str({ \
                        "fileName": args.fileName, \
                        "dictionary": args.dictionary, \
                        "sizeX": sizeX, \
                        "sizeY": sizeY, \
                        "squareLength": squareLength, \
                        "markerLength": markerLength, \
                        "borderBits": borderBits, \
                        "subSizeX": subSizeX, \
                        "subSizeY": subSizeY, \
                        "pageBorderX": pageBorderX, \
                        "pageBorderY": pageBorderY, \
                    }))

            subSize = None

            if(subSizeX > 0):
                if(subSizeY > 0):
                    subSize = (subSizeX, subSizeY)
                else:
                    subSize = (subSizeX, sizeY)
            else:
                if(subSizeY > 0):
                    subSize = (sizeX, subSizeY)
                else:
                    subSize = None

            # Gen
            MarkerPrinter.GenCharucoMarkerImage(args.fileName, args.dictionary, (sizeX, sizeY), squareLength, markerLength, borderBits=borderBits, subSize=subSize, pageBorder = (pageBorderX, pageBorderY))

    else:
        parser.print_help()
