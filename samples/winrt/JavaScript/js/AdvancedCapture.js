//// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
//// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
//// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
//// PARTICULAR PURPOSE.
////
//// Copyright (c) Microsoft Corporation. All rights reserved

(function () {
    "use strict";

    var cameraList = null;
    var mediaCaptureMgr = null;
    var captureInitSettings = null;

    var page = WinJS.UI.Pages.define("/html/AdvancedCapture.html", {

        ready: function (element, options) {
            scenarioInitialize();
        },

        unload: function (element, options) {
            // release resources
            releaseMediaCapture();
        }
    });

    function scenarioInitialize() {
        // Initialize the UI elements
        id("btnStartDevice").disabled = false;
        id("btnStartDevice").addEventListener("click", startDevice, false);
        id("btnStartPreview").disabled = true;
        id("videoEffect").disabled = true;
        id("btnStartPreview").addEventListener("click", startPreview, false);
        id("cameraSelect").addEventListener("change", onDeviceChange, false);

        id("videoEffect").addEventListener('change', addEffectToImageStream, false);

        enumerateCameras();
    }

    function initCameraSettings() {
        captureInitSettings = new Windows.Media.Capture.MediaCaptureInitializationSettings();
        captureInitSettings.streamingCaptureMode = Windows.Media.Capture.StreamingCaptureMode.video

        // If the user chose another capture device, use it by default
        var selectedIndex = id("cameraSelect").selectedIndex;
        var deviceInfo = cameraList[selectedIndex];
        captureInitSettings.videoDeviceId = deviceInfo.id;
    }

    // this function takes care of releasing the resources associated with media capturing
    function releaseMediaCapture() {
        if (mediaCaptureMgr) {
            mediaCaptureMgr.close();
            mediaCaptureMgr = null;
        }
    }

    //Initialize media capture with the current settings
    function startDevice() {
        displayStatus("Starting device");
        releaseMediaCapture();
        initCameraSettings();

        mediaCaptureMgr = new Windows.Media.Capture.MediaCapture();
        mediaCaptureMgr.initializeAsync(captureInitSettings).done(function (result) {
            // Update the UI
            id("btnStartPreview").disabled = false;
            id("btnStartDevice").disabled = true;
            displayStatus("Device started");
        });
    }

    function startPreview() {
        displayStatus("Starting preview");
        id("btnStartPreview").disabled = true;
        id("videoEffect").disabled = false;
        var video = id("previewVideo");
        video.src = URL.createObjectURL(mediaCaptureMgr, { oneTimeOnly: true });
        video.play();
        displayStatus("Preview started");
    }

    function addEffectToImageStream() {
        var effectId = id("videoEffect").selectedIndex;
        var props = new Windows.Foundation.Collections.PropertySet();
        props.insert("{698649BE-8EAE-4551-A4CB-3EC98FBD3D86}", effectId);

        mediaCaptureMgr.clearEffectsAsync(Windows.Media.Capture.MediaStreamType.videoPreview).then(function () {
            return mediaCaptureMgr.addEffectAsync(Windows.Media.Capture.MediaStreamType.videoPreview, 'OcvTransform.OcvImageManipulations', props);
        }).then(function () {
            displayStatus('Effect has been successfully added');
        }, errorHandler);
    }

    function enumerateCameras() {
        displayStatus("Enumerating capture devices");
        var cameraSelect = id("cameraSelect");
        cameraList = null;
        cameraList = new Array();

        // Clear the previous list of capture devices if any
        while (cameraSelect.length > 0) {
            cameraSelect.remove(0);
        }

        // Enumerate cameras and add them to the list
        var deviceInfo = Windows.Devices.Enumeration.DeviceInformation;
        deviceInfo.findAllAsync(Windows.Devices.Enumeration.DeviceClass.videoCapture).done(function (cameras) {
            if (cameras.length === 0) {
                cameraSelect.disabled = true;
                displayError("No camera was found");
                id("btnStartDevice").disabled = true;
                cameraSelect.add(new Option("No cameras available"));
            } else {
                cameras.forEach(function (camera) {
                    cameraList.push(camera);
                    cameraSelect.add(new Option(camera.name));
                });
            }
        }, errorHandler);
    }

    function onDeviceChange() {
        releaseMediaCapture();
        id("btnStartDevice").disabled = false;
        id("btnStartPreview").disabled = true;
        id("videoEffect").disabled = true;
        displayStatus("");
    }

    function suspendingHandler(suspendArg) {
        displayStatus("Suspended");
        releaseMediaCapture();
    }

    function resumingHandler(resumeArg) {
        displayStatus("Resumed");
        scenarioInitialize();
    }

    function errorHandler(err) {
        displayError(err.message);
    }

    function failedEventHandler(e) {
        displayError("Fatal error", e.message);
    }

    function displayStatus(statusText) {
        SdkSample.displayStatus(statusText);
    }

    function displayError(error) {
        SdkSample.displayError(error);
    }

    function id(elementId) {
        return document.getElementById(elementId);
    }
})();
