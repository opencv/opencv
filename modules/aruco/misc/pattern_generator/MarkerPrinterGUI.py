#!/usr/bin/env python3

# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2019, Josh Chien. All rights reserved.

from MarkerPrinter import *

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import time

import PIL.Image
import PIL.ImageTk

class MarkerPrinterGUI:

    def VisDPI(self, shape):
        scale0 = float(self.displayShape[0]) / float(shape[0])
        scale1 = float(self.displayShape[1]) / float(shape[1])
        if(scale0 > scale1):
            return scale1 * 96.0
        else:
            return scale0 * 96.0

    def OnShowingHelpGithub(self):
        messagebox.showinfo("Github",
            "https://github.com/dogod621/OpenCVMarkerPrinter")

    def OnCloseWindow(self):
        if(self.window is not None):
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                self.window.destroy()
                self.window = None

    def OnSelectCharucoMarkerDictionary(self, pDictName):
        self.charucoMarkerDictionaryStr.set(pDictName)

    def __SaveMarker(GenMarkerImageCallback, *args, **kwargs):

        if(kwargs.get("subSize",None) is not None):
            subSizeX, subSizeY = kwargs["subSize"]

            kwargs["subSize"] = None

            if(subSizeX > 0):
                if(subSizeY > 0):
                    kwargs["subSize"] = (subSizeX, subSizeY)
                else:
                    kwargs["subSize"] = (subSizeX, sizeY)
            else:
                if(subSizeY > 0):
                    kwargs["subSize"] = (sizeX, subSizeY)
                else:
                    kwargs["subSize"] = None

        try:
            askFileName = filedialog.asksaveasfilename(initialdir = os.path.abspath("./"), title = "Output", filetypes = (\
                ("scalable vector graphics files","*.svg"), \
                ("portable document format files","*.pdf"), \
                ("post script files","*.ps")),
                defaultextension="*.*")

            if (askFileName):
                GenMarkerImageCallback(askFileName, *args, **kwargs)

        except Exception as e:
            warnings.warn(str(e))
            messagebox.showinfo("Error", "Save marker failed")
            return

    def OnPreviewOrSaveCharucoMarker(self, askSave = False):
        try:
            sizeX = int(self.charucoMarkerChessboardSizeXStr.get())
            sizeY = int(self.charucoMarkerChessboardSizeYStr.get())
            squareLength = float(self.charucoMarkerSquareLengthStr.get())
            markerLength = float(self.charucoMarkerMarkerLengthStr.get())
            borderBits = int(self.charucoMarkerBorderBitsStr.get())
            dictionary = self.charucoMarkerDictionaryStr.get()
            subSizeX = int(self.charucoMarkerSaveSubSizeXStr.get())
            subSizeY = int(self.charucoMarkerSaveSubSizeYStr.get())
            pageBorderX = float(self.charucoMarkerSavePageBorderXStr.get())
            pageBorderY = float(self.charucoMarkerSavePageBorderYStr.get())
        except ValueError as e:
            warnings.warn(str(e))
            messagebox.showinfo("Error", "Enter invalid parameters")
            return
        except Exception as e:
            warnings.warn(str(e))
            messagebox.showinfo("Error", "Fail to get parameters")
            return

        # Preview
        try:
            dpi = self.VisDPI(((sizeY * squareLength + pageBorderY * 2) * MarkerPrinter.ptPerMeter, (sizeX * squareLength + pageBorderX * 2) * MarkerPrinter.ptPerMeter))
            tkImage = PIL.ImageTk.PhotoImage(image = MarkerPrinter.PreviewCharucoMarkerImage(dictionary, (sizeX, sizeY), squareLength, markerLength, borderBits=borderBits, pageBorder = (pageBorderX, pageBorderY), dpi=dpi))
            self.charucoMarkerImageLabel.imgtk = tkImage
            self.charucoMarkerImageLabel.config(image=tkImage)
        except Exception as e:
            warnings.warn(str(e))
            messagebox.showinfo("Error", "create marker failed")
            return

        # Save
        if(askSave):
            MarkerPrinterGUI.__SaveMarker(MarkerPrinter.GenCharucoMarkerImage, \
                dictionary, (sizeX, sizeY), squareLength, markerLength, borderBits=borderBits, subSize = (subSizeX, subSizeY), pageBorder = (pageBorderX, pageBorderY))

    def OnPreviewCharucoMarker(self):
        self.OnPreviewOrSaveCharucoMarker(askSave = False)

    def OnSaveCharucoMarker(self):
        self.OnPreviewOrSaveCharucoMarker(askSave = True)

    def InitCharucoMarkerTab(self):
        self.charucoMarkerUIFrame = ttk.Frame(self.charucoMarkerTab)
        self.charucoMarkerImageTab = ttk.Frame(self.charucoMarkerTab)
        self.charucoMarkerUIFrame2 = ttk.Frame(self.charucoMarkerTab)

        self.charucoMarkerUIFrame.grid(row=0, column=0, sticky = tk.NSEW)
        self.charucoMarkerImageTab.grid(row=1, column=0, sticky = tk.NSEW)
        self.charucoMarkerUIFrame2.grid(row=2, column=0, sticky = tk.NSEW)

        self.charucoMarkerImageLabel = tk.Label(self.charucoMarkerImageTab)
        self.charucoMarkerImageLabel.grid(row=0, column=0, sticky = tk.NSEW)

        tk.Label(self.charucoMarkerUIFrame, text="dictionary").grid(row=0, column=0, sticky = tk.NSEW)
        tk.Label(self.charucoMarkerUIFrame, text="chessboardSizeX").grid(row=0, column=1, sticky = tk.NSEW)
        tk.Label(self.charucoMarkerUIFrame, text="chessboardSizeY").grid(row=0, column=2, sticky = tk.NSEW)
        tk.Label(self.charucoMarkerUIFrame, text="squareLength (Unit: Meter)").grid(row=0, column=3, sticky = tk.NSEW)
        tk.Label(self.charucoMarkerUIFrame, text="markerLength (Unit: Meter)").grid(row=0, column=4, sticky = tk.NSEW)
        tk.Label(self.charucoMarkerUIFrame, text="borderBits").grid(row=0, column=5, sticky = tk.NSEW)

        self.charucoMarkerDictionaryStr = tk.StringVar()
        self.charucoMarkerChessboardSizeXStr = tk.StringVar()
        self.charucoMarkerChessboardSizeXStr.set("16")
        self.charucoMarkerChessboardSizeYStr = tk.StringVar()
        self.charucoMarkerChessboardSizeYStr.set("9")
        self.charucoMarkerSquareLengthStr = tk.StringVar()
        self.charucoMarkerSquareLengthStr.set("0.09")
        self.charucoMarkerMarkerLengthStr = tk.StringVar()
        self.charucoMarkerMarkerLengthStr.set("0.07")
        self.charucoMarkerBorderBitsStr = tk.StringVar()
        self.charucoMarkerBorderBitsStr.set("1")

        self.charucoMarkerDictionaryMenue = tk.OptionMenu(self.charucoMarkerUIFrame, self.charucoMarkerDictionaryStr, "DICT_ARUCO_ORIGINAL", command = self.OnSelectCharucoMarkerDictionary)
        self.charucoMarkerDictionaryMenue.grid(row=1, column=0, sticky = tk.NSEW)
        tk.Entry(self.charucoMarkerUIFrame, textvariable=self.charucoMarkerChessboardSizeXStr).grid(row=1, column=1, sticky = tk.NSEW)
        tk.Entry(self.charucoMarkerUIFrame, textvariable=self.charucoMarkerChessboardSizeYStr).grid(row=1, column=2, sticky = tk.NSEW)
        tk.Entry(self.charucoMarkerUIFrame, textvariable=self.charucoMarkerSquareLengthStr).grid(row=1, column=3, sticky = tk.NSEW)
        tk.Entry(self.charucoMarkerUIFrame, textvariable=self.charucoMarkerMarkerLengthStr).grid(row=1, column=4, sticky = tk.NSEW)
        tk.Entry(self.charucoMarkerUIFrame, textvariable=self.charucoMarkerBorderBitsStr).grid(row=1, column=5, sticky = tk.NSEW)

        tk.Button(self.charucoMarkerUIFrame2, text = "Preview", command = self.OnPreviewCharucoMarker).grid(row=1, column=0, sticky = tk.NSEW)
        tk.Button(self.charucoMarkerUIFrame2, text = "Save", command = self.OnSaveCharucoMarker).grid(row=1, column=1, sticky = tk.NSEW)

        tk.Label(self.charucoMarkerUIFrame2, text="Save options:").grid(row=0, column=2, sticky = tk.NSEW)
        tk.Label(self.charucoMarkerUIFrame2, text="(set 0 as disable)").grid(row=1, column=2, sticky = tk.NSEW)
        tk.Label(self.charucoMarkerUIFrame2, text="subSizeX").grid(row=0, column=3, sticky = tk.NSEW)
        tk.Label(self.charucoMarkerUIFrame2, text="subSizeY").grid(row=0, column=4, sticky = tk.NSEW)
        tk.Label(self.charucoMarkerUIFrame2, text="Divide to chunks, chunk sizeX").grid(row=2, column=3, sticky = tk.NSEW)
        tk.Label(self.charucoMarkerUIFrame2, text="Divide to chunks, chunk sizeY").grid(row=2, column=4, sticky = tk.NSEW)
        tk.Label(self.charucoMarkerUIFrame2, text="pageBorderX (Unit: Meter)").grid(row=0, column=5, sticky = tk.NSEW)
        tk.Label(self.charucoMarkerUIFrame2, text="pageBorderY (Unit: Meter)").grid(row=0, column=6, sticky = tk.NSEW)
        tk.Label(self.charucoMarkerUIFrame2, text="Border or page").grid(row=2, column=5, sticky = tk.NSEW)
        tk.Label(self.charucoMarkerUIFrame2, text="Border or page").grid(row=2, column=6, sticky = tk.NSEW)

        self.charucoMarkerSaveSubSizeXStr = tk.StringVar()
        self.charucoMarkerSaveSubSizeXStr.set("0")
        self.charucoMarkerSaveSubSizeYStr = tk.StringVar()
        self.charucoMarkerSaveSubSizeYStr.set("0")
        self.charucoMarkerSavePageBorderXStr = tk.StringVar()
        self.charucoMarkerSavePageBorderXStr.set("0.02")
        self.charucoMarkerSavePageBorderYStr = tk.StringVar()
        self.charucoMarkerSavePageBorderYStr.set("0.02")

        tk.Entry(self.charucoMarkerUIFrame2, textvariable=self.charucoMarkerSaveSubSizeXStr).grid(row=1, column=3, sticky = tk.NSEW)
        tk.Entry(self.charucoMarkerUIFrame2, textvariable=self.charucoMarkerSaveSubSizeYStr).grid(row=1, column=4, sticky = tk.NSEW)
        tk.Entry(self.charucoMarkerUIFrame2, textvariable=self.charucoMarkerSavePageBorderXStr).grid(row=1, column=5, sticky = tk.NSEW)
        tk.Entry(self.charucoMarkerUIFrame2, textvariable=self.charucoMarkerSavePageBorderYStr).grid(row=1, column=6, sticky = tk.NSEW)

        self.charucoMarkerDictionaryMenue['menu'].delete(0, 'end')
        for dictName in self.dictList:
            self.charucoMarkerDictionaryMenue['menu'].add_command(label=dictName, command=tk._setit(self.charucoMarkerDictionaryStr, dictName, self.OnSelectCharucoMarkerDictionary))

        self.OnSelectCharucoMarkerDictionary("DICT_ARUCO_ORIGINAL")

    def OnSelectArucoGridMarkerDictionary(self, pDictName):
        self.arucoGridMarkerDictionaryStr.set(pDictName)

    def OnPreviewOrSaveArucoGridMarker(self, askSave = False):
        try:
            markersX = int(self.arucoGridMarkerMarkersXStr.get())
            markersY = int(self.arucoGridMarkerMarkersYStr.get())
            markerLength = float(self.arucoGridMarkerMarkerLengthStr.get())
            markerSeparation = float(self.arucoGridMarkerMarkerSeparationStr.get())
            borderBits = int(self.arucoGridMarkerBorderBitsStr.get())
            firstMarker = int(self.arucoGridMarkerFirstMarkerStr.get())
            dictionary = self.arucoGridMarkerDictionaryStr.get()
            subSizeX = int(self.arucoGridMarkerSaveSubSizeXStr.get())
            subSizeY = int(self.arucoGridMarkerSaveSubSizeYStr.get())
            pageBorderX = float(self.arucoGridMarkerSavePageBorderXStr.get())
            pageBorderY = float(self.arucoGridMarkerSavePageBorderYStr.get())
        except ValueError as e:
            warnings.warn(str(e))
            messagebox.showinfo("Error", "Enter invalid parameters")
            return
        except Exception as e:
            warnings.warn(str(e))
            messagebox.showinfo("Error", "Fail to get parameters")
            return

        # Preview
        try:
            dpi=self.VisDPI(((markersY * markerLength + (markersY  - 1) * markerSeparation + pageBorderY * 2) * MarkerPrinter.ptPerMeter, (markersX * markerLength + (markersX  - 1) * markerSeparation + pageBorderX * 2) * MarkerPrinter.ptPerMeter))
            tkImage = PIL.ImageTk.PhotoImage(image = MarkerPrinter.PreviewArucoGridMarkerImage(dictionary, (markersX, markersY), markerLength, markerSeparation, firstMarker, borderBits=borderBits, pageBorder = (pageBorderX, pageBorderY), dpi=dpi))
            self.arucoGridMarkerImageLabel.imgtk = tkImage
            self.arucoGridMarkerImageLabel.config(image=tkImage)
        except Exception as e:
            warnings.warn(str(e))
            messagebox.showinfo("Error", "create marker failed")
            return

        # Save
        if(askSave):
            MarkerPrinterGUI.__SaveMarker(MarkerPrinter.GenArucoGridMarkerImage, \
                dictionary, (markersX, markersY), markerLength, markerSeparation, firstMarker, borderBits=borderBits, subSize = (subSizeX, subSizeY), pageBorder = (pageBorderX, pageBorderY))

    def OnPreviewArucoGridMarker(self):
        self.OnPreviewOrSaveArucoGridMarker(askSave = False)

    def OnSaveArucoGridMarker(self):
        self.OnPreviewOrSaveArucoGridMarker(askSave = True)

    def InitArucoGridMarkerTab(self):
        self.arucoGridMarkerUIFrame = ttk.Frame(self.arucoGridMarkerTab)
        self.arucoGridMarkerImageTab = ttk.Frame(self.arucoGridMarkerTab)
        self.arucoGridMarkerUIFrame2 = ttk.Frame(self.arucoGridMarkerTab)

        self.arucoGridMarkerUIFrame.grid(row=0, column=0, sticky = tk.NSEW)
        self.arucoGridMarkerImageTab.grid(row=1, column=0, sticky = tk.NSEW)
        self.arucoGridMarkerUIFrame2.grid(row=2, column=0, sticky = tk.NSEW)

        self.arucoGridMarkerImageLabel = tk.Label(self.arucoGridMarkerImageTab)
        self.arucoGridMarkerImageLabel.grid(row=0, column=0, sticky = tk.NSEW)

        tk.Label(self.arucoGridMarkerUIFrame, text="dictionary").grid(row=0, column=0, sticky = tk.NSEW)
        tk.Label(self.arucoGridMarkerUIFrame, text="markersX").grid(row=0, column=1, sticky = tk.NSEW)
        tk.Label(self.arucoGridMarkerUIFrame, text="markersY").grid(row=0, column=2, sticky = tk.NSEW)
        tk.Label(self.arucoGridMarkerUIFrame, text="markerLength (Unit: Meter)").grid(row=0, column=3, sticky = tk.NSEW)
        tk.Label(self.arucoGridMarkerUIFrame, text="markerSeparation (Unit: Meter)").grid(row=0, column=4, sticky = tk.NSEW)
        tk.Label(self.arucoGridMarkerUIFrame, text="firstMarker").grid(row=0, column=5, sticky = tk.NSEW)
        tk.Label(self.arucoGridMarkerUIFrame, text="borderBits").grid(row=0, column=6, sticky = tk.NSEW)

        self.arucoGridMarkerDictionaryStr = tk.StringVar()
        self.arucoGridMarkerMarkersXStr = tk.StringVar()
        self.arucoGridMarkerMarkersXStr.set("16")
        self.arucoGridMarkerMarkersYStr = tk.StringVar()
        self.arucoGridMarkerMarkersYStr.set("9")
        self.arucoGridMarkerMarkerLengthStr = tk.StringVar()
        self.arucoGridMarkerMarkerLengthStr.set("0.07")
        self.arucoGridMarkerMarkerSeparationStr = tk.StringVar()
        self.arucoGridMarkerMarkerSeparationStr.set("0.02")
        self.arucoGridMarkerFirstMarkerStr = tk.StringVar()
        self.arucoGridMarkerFirstMarkerStr.set("0")
        self.arucoGridMarkerBorderBitsStr = tk.StringVar()
        self.arucoGridMarkerBorderBitsStr.set("1")

        self.arucoGridMarkerDictionaryMenue = tk.OptionMenu(self.arucoGridMarkerUIFrame, self.arucoGridMarkerDictionaryStr, "DICT_ARUCO_ORIGINAL", command = self.OnSelectArucoGridMarkerDictionary)
        self.arucoGridMarkerDictionaryMenue.grid(row=1, column=0, sticky = tk.NSEW)
        tk.Entry(self.arucoGridMarkerUIFrame, textvariable=self.arucoGridMarkerMarkersXStr).grid(row=1, column=1, sticky = tk.NSEW)
        tk.Entry(self.arucoGridMarkerUIFrame, textvariable=self.arucoGridMarkerMarkersYStr).grid(row=1, column=2, sticky = tk.NSEW)
        tk.Entry(self.arucoGridMarkerUIFrame, textvariable=self.arucoGridMarkerMarkerLengthStr).grid(row=1, column=3, sticky = tk.NSEW)
        tk.Entry(self.arucoGridMarkerUIFrame, textvariable=self.arucoGridMarkerMarkerSeparationStr).grid(row=1, column=4, sticky = tk.NSEW)
        tk.Entry(self.arucoGridMarkerUIFrame, textvariable=self.arucoGridMarkerFirstMarkerStr).grid(row=1, column=5, sticky = tk.NSEW)
        tk.Entry(self.arucoGridMarkerUIFrame, textvariable=self.arucoGridMarkerBorderBitsStr).grid(row=1, column=6, sticky = tk.NSEW)

        tk.Button(self.arucoGridMarkerUIFrame2, text = "Preview", command = self.OnPreviewArucoGridMarker).grid(row=1, column=0, sticky = tk.NSEW)
        tk.Button(self.arucoGridMarkerUIFrame2, text = "Save", command = self.OnSaveArucoGridMarker).grid(row=1, column=1, sticky = tk.NSEW)

        tk.Label(self.arucoGridMarkerUIFrame2, text="Save options:").grid(row=0, column=2, sticky = tk.NSEW)
        tk.Label(self.arucoGridMarkerUIFrame2, text="(set 0 as disable)").grid(row=1, column=2, sticky = tk.NSEW)
        tk.Label(self.arucoGridMarkerUIFrame2, text="subSizeX").grid(row=0, column=3, sticky = tk.NSEW)
        tk.Label(self.arucoGridMarkerUIFrame2, text="subSizeY").grid(row=0, column=4, sticky = tk.NSEW)
        tk.Label(self.arucoGridMarkerUIFrame2, text="Divide to chunks, chunk sizeX").grid(row=2, column=3, sticky = tk.NSEW)
        tk.Label(self.arucoGridMarkerUIFrame2, text="Divide to chunks, chunk sizeY").grid(row=2, column=4, sticky = tk.NSEW)
        tk.Label(self.arucoGridMarkerUIFrame2, text="pageBorderX (Unit: Meter)").grid(row=0, column=5, sticky = tk.NSEW)
        tk.Label(self.arucoGridMarkerUIFrame2, text="pageBorderY (Unit: Meter)").grid(row=0, column=6, sticky = tk.NSEW)
        tk.Label(self.arucoGridMarkerUIFrame2, text="Border or page").grid(row=2, column=5, sticky = tk.NSEW)
        tk.Label(self.arucoGridMarkerUIFrame2, text="Border or page").grid(row=2, column=6, sticky = tk.NSEW)

        self.arucoGridMarkerSaveSubSizeXStr = tk.StringVar()
        self.arucoGridMarkerSaveSubSizeXStr.set("0")
        self.arucoGridMarkerSaveSubSizeYStr = tk.StringVar()
        self.arucoGridMarkerSaveSubSizeYStr.set("0")
        self.arucoGridMarkerSavePageBorderXStr = tk.StringVar()
        self.arucoGridMarkerSavePageBorderXStr.set("0.02")
        self.arucoGridMarkerSavePageBorderYStr = tk.StringVar()
        self.arucoGridMarkerSavePageBorderYStr.set("0.02")

        tk.Entry(self.arucoGridMarkerUIFrame2, textvariable=self.arucoGridMarkerSaveSubSizeXStr).grid(row=1, column=3, sticky = tk.NSEW)
        tk.Entry(self.arucoGridMarkerUIFrame2, textvariable=self.arucoGridMarkerSaveSubSizeYStr).grid(row=1, column=4, sticky = tk.NSEW)
        tk.Entry(self.arucoGridMarkerUIFrame2, textvariable=self.arucoGridMarkerSavePageBorderXStr).grid(row=1, column=5, sticky = tk.NSEW)
        tk.Entry(self.arucoGridMarkerUIFrame2, textvariable=self.arucoGridMarkerSavePageBorderYStr).grid(row=1, column=6, sticky = tk.NSEW)

        self.arucoGridMarkerDictionaryMenue['menu'].delete(0, 'end')
        for dictName in self.dictList:
            self.arucoGridMarkerDictionaryMenue['menu'].add_command(label=dictName, command=tk._setit(self.arucoGridMarkerDictionaryStr, dictName, self.OnSelectArucoGridMarkerDictionary))

        self.OnSelectArucoGridMarkerDictionary("DICT_ARUCO_ORIGINAL")

    def OnSelectArucoMarkerDictionary(self, pDictName):
        self.arucoMarkerDictionaryStr.set(pDictName)

    def OnPreviewOrSaveArucoMarker(self, askSave = False):
        try:
            markerID = int(self.arucoMarkerMarkerIDStr.get())
            markerLength = float(self.arucoMarkerMarkerLengthStr.get())
            borderBits = int(self.arucoMarkerBorderBitsStr.get())
            dictionary = self.arucoMarkerDictionaryStr.get()
            pageBorderX = float(self.arucoMarkerSavePageBorderXStr.get())
            pageBorderY = float(self.arucoMarkerSavePageBorderYStr.get())
        except ValueError as e:
            warnings.warn(str(e))
            messagebox.showinfo("Error", "Enter invalid parameters")
            return
        except Exception as e:
            warnings.warn(str(e))
            messagebox.showinfo("Error", "Fail to get parameters")
            return

        # Preview
        try:
            dpi=self.VisDPI(((markerLength  + pageBorderY * 2) * MarkerPrinter.ptPerMeter, (markerLength + pageBorderX * 2) * MarkerPrinter.ptPerMeter))
            tkImage = PIL.ImageTk.PhotoImage(image = MarkerPrinter.PreviewArucoMarkerImage(dictionary, markerID, markerLength, borderBits=borderBits, pageBorder = (pageBorderX, pageBorderY), dpi=dpi))
            self.arucoMarkerImageLabel.imgtk = tkImage
            self.arucoMarkerImageLabel.config(image=tkImage)
        except Exception as e:
            warnings.warn(str(e))
            messagebox.showinfo("Error", "create marker failed")
            return

        # Save
        if(askSave):
            MarkerPrinterGUI.__SaveMarker(MarkerPrinter.GenArucoMarkerImage, \
                dictionary, markerID, markerLength, borderBits=borderBits, pageBorder = (pageBorderX, pageBorderY))

    def OnPreviewArucoMarker(self):
        self.OnPreviewOrSaveArucoMarker(askSave = False)

    def OnSaveArucoMarker(self):
        self.OnPreviewOrSaveArucoMarker(askSave = True)

    def InitArucoMarkerTab(self):
        self.arucoMarkerUIFrame = ttk.Frame(self.arucoMarkerTab)
        self.arucoMarkerImageTab = ttk.Frame(self.arucoMarkerTab)
        self.arucoMarkerUIFrame2 = ttk.Frame(self.arucoMarkerTab)

        self.arucoMarkerUIFrame.grid(row=0, column=0, sticky = tk.NSEW)
        self.arucoMarkerImageTab.grid(row=1, column=0, sticky = tk.NSEW)
        self.arucoMarkerUIFrame2.grid(row=2, column=0, sticky = tk.NSEW)

        self.arucoMarkerImageLabel = tk.Label(self.arucoMarkerImageTab)
        self.arucoMarkerImageLabel.grid(row=0, column=0, sticky = tk.NSEW)

        tk.Label(self.arucoMarkerUIFrame, text="dictionary").grid(row=0, column=0, sticky = tk.NSEW)
        tk.Label(self.arucoMarkerUIFrame, text="markerID").grid(row=0, column=1, sticky = tk.NSEW)
        tk.Label(self.arucoMarkerUIFrame, text="markerLength (Unit: Meter)").grid(row=0, column=2, sticky = tk.NSEW)
        tk.Label(self.arucoMarkerUIFrame, text="borderBits").grid(row=0, column=3, sticky = tk.NSEW)

        self.arucoMarkerDictionaryStr = tk.StringVar()
        self.arucoMarkerMarkerIDStr = tk.StringVar()
        self.arucoMarkerMarkerIDStr.set("0")
        self.arucoMarkerMarkerLengthStr = tk.StringVar()
        self.arucoMarkerMarkerLengthStr.set("0.07")
        self.arucoMarkerBorderBitsStr = tk.StringVar()
        self.arucoMarkerBorderBitsStr.set("1")

        self.arucoMarkerDictionaryMenue = tk.OptionMenu(self.arucoMarkerUIFrame, self.arucoMarkerDictionaryStr, "DICT_ARUCO_ORIGINAL", command = self.OnSelectArucoMarkerDictionary)
        self.arucoMarkerDictionaryMenue.grid(row=1, column=0, sticky = tk.NSEW)
        tk.Entry(self.arucoMarkerUIFrame, textvariable=self.arucoMarkerMarkerIDStr).grid(row=1, column=1, sticky = tk.NSEW)
        tk.Entry(self.arucoMarkerUIFrame, textvariable=self.arucoMarkerMarkerLengthStr).grid(row=1, column=2, sticky = tk.NSEW)
        tk.Entry(self.arucoMarkerUIFrame, textvariable=self.arucoMarkerBorderBitsStr).grid(row=1, column=3, sticky = tk.NSEW)

        tk.Button(self.arucoMarkerUIFrame2, text = "Preview", command = self.OnPreviewArucoMarker).grid(row=0, column=0, sticky = tk.NSEW)
        tk.Button(self.arucoMarkerUIFrame2, text = "Save", command = self.OnSaveArucoMarker).grid(row=0, column=1, sticky = tk.NSEW)

        tk.Label(self.arucoMarkerUIFrame2, text="Save options:").grid(row=0, column=2, sticky = tk.NSEW)
        tk.Label(self.arucoMarkerUIFrame2, text="(set 0 as disable)").grid(row=1, column=2, sticky = tk.NSEW)
        tk.Label(self.arucoMarkerUIFrame2, text="pageBorderX (Unit: Meter)").grid(row=0, column=3, sticky = tk.NSEW)
        tk.Label(self.arucoMarkerUIFrame2, text="pageBorderY (Unit: Meter)").grid(row=0, column=4, sticky = tk.NSEW)
        tk.Label(self.arucoMarkerUIFrame2, text="Border or page").grid(row=2, column=3, sticky = tk.NSEW)
        tk.Label(self.arucoMarkerUIFrame2, text="Border or page").grid(row=2, column=4, sticky = tk.NSEW)

        self.arucoMarkerSavePageBorderXStr = tk.StringVar()
        self.arucoMarkerSavePageBorderXStr.set("0.02")
        self.arucoMarkerSavePageBorderYStr = tk.StringVar()
        self.arucoMarkerSavePageBorderYStr.set("0.02")

        tk.Entry(self.arucoMarkerUIFrame2, textvariable=self.arucoMarkerSavePageBorderXStr).grid(row=1, column=3, sticky = tk.NSEW)
        tk.Entry(self.arucoMarkerUIFrame2, textvariable=self.arucoMarkerSavePageBorderYStr).grid(row=1, column=4, sticky = tk.NSEW)

        self.arucoMarkerDictionaryMenue['menu'].delete(0, 'end')
        for dictName in self.dictList:
            self.arucoMarkerDictionaryMenue['menu'].add_command(label=dictName, command=tk._setit(self.arucoMarkerDictionaryStr, dictName, self.OnSelectArucoMarkerDictionary))

        self.OnSelectArucoMarkerDictionary("DICT_ARUCO_ORIGINAL")

    def OnPreviewOrSaveChessMarker(self, askSave = False):
        try:
            sizeX = int(self.chessMarkerChessboardSizeXStr.get())
            sizeY = int(self.chessMarkerChessboardSizeYStr.get())
            squareLength = float(self.chessMarkerSquareLengthStr.get())
            subSizeX = int(self.chessMarkerSaveSubSizeXStr.get())
            subSizeY = int(self.chessMarkerSaveSubSizeYStr.get())
            pageBorderX = float(self.chessMarkerSavePageBorderXStr.get())
            pageBorderY = float(self.chessMarkerSavePageBorderYStr.get())
        except ValueError as e:
            warnings.warn(str(e))
            messagebox.showinfo("Error", "Enter invalid parameters")
            return
        except Exception as e:
            warnings.warn(str(e))
            messagebox.showinfo("Error", "Fail to get parameters")
            return

        # Preview
        try:
            dpi=self.VisDPI(((sizeY * squareLength + pageBorderY * 2) * MarkerPrinter.ptPerMeter, (sizeX * squareLength + pageBorderX * 2) * MarkerPrinter.ptPerMeter))
            tkImage = PIL.ImageTk.PhotoImage(image = MarkerPrinter.PreviewChessMarkerImage((sizeX, sizeY), squareLength, pageBorder = (pageBorderX, pageBorderY), dpi=dpi))
            self.chessMarkerImageLabel.imgtk = tkImage
            self.chessMarkerImageLabel.config(image=tkImage)
        except Exception as e:
            warnings.warn(str(e))
            messagebox.showinfo("Error", "create marker failed")
            return

        # Save
        if(askSave):
            MarkerPrinterGUI.__SaveMarker(MarkerPrinter.GenChessMarkerImage, \
                (sizeX, sizeY), squareLength, subSize = (subSizeX, subSizeY), pageBorder = (pageBorderX, pageBorderY))

    def OnPreviewChessMarker(self):
        self.OnPreviewOrSaveChessMarker(askSave = False)

    def OnSaveChessMarker(self):
        self.OnPreviewOrSaveChessMarker(askSave = True)

    def InitChessMarkerTab(self):
        self.chessMarkerUIFrame = ttk.Frame(self.chessMarkerTab)
        self.chessMarkerImageTab = ttk.Frame(self.chessMarkerTab)
        self.chessMarkerUIFrame2 = ttk.Frame(self.chessMarkerTab)

        self.chessMarkerUIFrame.grid(row=0, column=0, sticky = tk.NSEW)
        self.chessMarkerImageTab.grid(row=1, column=0, sticky = tk.NSEW)
        self.chessMarkerUIFrame2.grid(row=2, column=0, sticky = tk.NSEW)

        self.chessMarkerImageLabel = tk.Label(self.chessMarkerImageTab)
        self.chessMarkerImageLabel.grid(row=0, column=0, sticky = tk.NSEW)

        tk.Label(self.chessMarkerUIFrame, text="chessboardSizeX").grid(row=0, column=0, sticky = tk.NSEW)
        tk.Label(self.chessMarkerUIFrame, text="chessboardSizeY").grid(row=0, column=1, sticky = tk.NSEW)
        tk.Label(self.chessMarkerUIFrame, text="squareLength (Unit: Meter)").grid(row=0, column=2, sticky = tk.NSEW)

        self.chessMarkerChessboardSizeXStr = tk.StringVar()
        self.chessMarkerChessboardSizeXStr.set("16")
        self.chessMarkerChessboardSizeYStr = tk.StringVar()
        self.chessMarkerChessboardSizeYStr.set("9")
        self.chessMarkerSquareLengthStr = tk.StringVar()
        self.chessMarkerSquareLengthStr.set("0.09")

        tk.Entry(self.chessMarkerUIFrame, textvariable=self.chessMarkerChessboardSizeXStr).grid(row=1, column=0, sticky = tk.NSEW)
        tk.Entry(self.chessMarkerUIFrame, textvariable=self.chessMarkerChessboardSizeYStr).grid(row=1, column=1, sticky = tk.NSEW)
        tk.Entry(self.chessMarkerUIFrame, textvariable=self.chessMarkerSquareLengthStr).grid(row=1, column=2, sticky = tk.NSEW)

        tk.Button(self.chessMarkerUIFrame2, text = "Preview", command = self.OnPreviewChessMarker).grid(row=1, column=0, sticky = tk.NSEW)
        tk.Button(self.chessMarkerUIFrame2, text = "Save", command = self.OnSaveChessMarker).grid(row=1, column=1, sticky = tk.NSEW)

        tk.Label(self.chessMarkerUIFrame2, text="Save options:").grid(row=0, column=2, sticky = tk.NSEW)
        tk.Label(self.chessMarkerUIFrame2, text="(set 0 as disable)").grid(row=1, column=2, sticky = tk.NSEW)
        tk.Label(self.chessMarkerUIFrame2, text="subSizeX").grid(row=0, column=3, sticky = tk.NSEW)
        tk.Label(self.chessMarkerUIFrame2, text="subSizeY").grid(row=0, column=4, sticky = tk.NSEW)
        tk.Label(self.chessMarkerUIFrame2, text="Divide to chunks, chunk sizeX").grid(row=2, column=3, sticky = tk.NSEW)
        tk.Label(self.chessMarkerUIFrame2, text="Divide to chunks, chunk sizeY").grid(row=2, column=4, sticky = tk.NSEW)
        tk.Label(self.chessMarkerUIFrame2, text="pageBorderX (Unit: Meter)").grid(row=0, column=5, sticky = tk.NSEW)
        tk.Label(self.chessMarkerUIFrame2, text="pageBorderY (Unit: Meter)").grid(row=0, column=6, sticky = tk.NSEW)
        tk.Label(self.chessMarkerUIFrame2, text="Border or page").grid(row=2, column=5, sticky = tk.NSEW)
        tk.Label(self.chessMarkerUIFrame2, text="Border or page").grid(row=2, column=6, sticky = tk.NSEW)

        self.chessMarkerSaveSubSizeXStr = tk.StringVar()
        self.chessMarkerSaveSubSizeXStr.set("0")
        self.chessMarkerSaveSubSizeYStr = tk.StringVar()
        self.chessMarkerSaveSubSizeYStr.set("0")
        self.chessMarkerSavePageBorderXStr = tk.StringVar()
        self.chessMarkerSavePageBorderXStr.set("0.02")
        self.chessMarkerSavePageBorderYStr = tk.StringVar()
        self.chessMarkerSavePageBorderYStr.set("0.02")

        tk.Entry(self.chessMarkerUIFrame2, textvariable=self.chessMarkerSaveSubSizeXStr).grid(row=1, column=3, sticky = tk.NSEW)
        tk.Entry(self.chessMarkerUIFrame2, textvariable=self.chessMarkerSaveSubSizeYStr).grid(row=1, column=4, sticky = tk.NSEW)
        tk.Entry(self.chessMarkerUIFrame2, textvariable=self.chessMarkerSavePageBorderXStr).grid(row=1, column=5, sticky = tk.NSEW)
        tk.Entry(self.chessMarkerUIFrame2, textvariable=self.chessMarkerSavePageBorderYStr).grid(row=1, column=6, sticky = tk.NSEW)

    def Update(self):
        time.sleep(0)
        self.window.after(self.delay, self.Update)

    def __init__(self, pDelay=15, pDisplayShape=(int(400), int(1200))):
        self.delay = pDelay
        self.displayShape = pDisplayShape

        self.dictList = MarkerPrinter.arucoDictBytesList.keys()

        # GUI
        self.window = tk.Tk()
        self.notebook = ttk.Notebook(self.window)
        self.notebook.grid(row=0, column=0, sticky = tk.NSEW)

        self.window.title("MarkerPrinterGUI")
        self.window.config(cursor="arrow")
        self.window.protocol("WM_DELETE_WINDOW", self.OnCloseWindow)

        # Menues
        self.menu = tk.Menu(self.window)
        self.helpMenu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Help", menu=self.helpMenu)
        self.helpMenu.add_command(label="Github", command=self.OnShowingHelpGithub)
        self.helpMenu.add_command(label="DEBUG_LINE_MODE", command=self.On_DEBUG_LINE_MODE)
        self.helpMenu.add_command(label="DEBUG_BLOCK_MODE", command=self.On_DEBUG_BLOCK_MODE)
        self.helpMenu.add_command(label="CLOSE_DEBUG_MODE", command=self.On_CLOSE_DEBUG_MODE)
        self.window.config(menu=self.menu)

        self.charucoMarkerTab = ttk.Frame(self.notebook)
        self.arucoMarkerTab = ttk.Frame(self.notebook)
        self.arucoGridMarkerTab = ttk.Frame(self.notebook)
        self.chessMarkerTab = ttk.Frame(self.notebook)

        self.notebook.add(self.charucoMarkerTab, text='ChArUco Marker')
        self.notebook.add(self.arucoMarkerTab, text='ArUco Marker')
        self.notebook.add(self.arucoGridMarkerTab, text='ArUcoGrid Marker')
        self.notebook.add(self.chessMarkerTab, text='Chessboard Marker')

        self.InitCharucoMarkerTab()
        self.InitArucoMarkerTab()
        self.InitArucoGridMarkerTab()
        self.InitChessMarkerTab()

        self.Update()
        self.window.mainloop()

    def On_DEBUG_LINE_MODE(self):
        messagebox.showinfo("Note", "You enabled the debug mode: \"LINE\"")
        MarkerPrinter.debugMode = "LINE"

    def On_DEBUG_BLOCK_MODE(self):
        messagebox.showinfo("Note", "You enabled the debug mode: \"BLOCK\"")
        MarkerPrinter.debugMode = "BLOCK"

    def On_CLOSE_DEBUG_MODE(self):
        messagebox.showinfo("Note", "You closed the debug mode")
        MarkerPrinter.debugMode = None

if __name__ == '__main__':
    MarkerPrinterGUI()
