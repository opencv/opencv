//// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
//// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
//// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
//// PARTICULAR PURPOSE.
////
//// Copyright (c) Microsoft Corporation. All rights reserved


(function () {
    "use strict";

    var sampleTitle = "OpenCV Image Manipulations sample";

    var scenarios = [
        { url: "/html/AdvancedCapture.html", title: "Enumerate cameras and add a video effect" },
    ];

    function activated(eventObject) {
        if (eventObject.detail.kind === Windows.ApplicationModel.Activation.ActivationKind.launch) {
            // Use setPromise to indicate to the system that the splash screen must not be torn down
            // until after processAll and navigate complete asynchronously.
            eventObject.setPromise(WinJS.UI.processAll().then(function () {
                // Navigate to either the first scenario or to the last running scenario
                // before suspension or termination.
                var url = WinJS.Application.sessionState.lastUrl || scenarios[0].url;
                return WinJS.Navigation.navigate(url);
            }));
        }
    }

    WinJS.Navigation.addEventListener("navigated", function (eventObject) {
        var url = eventObject.detail.location;
        var host = document.getElementById("contentHost");
        // Call unload method on current scenario, if there is one
        host.winControl && host.winControl.unload && host.winControl.unload();
        WinJS.Utilities.empty(host);
        eventObject.detail.setPromise(WinJS.UI.Pages.render(url, host, eventObject.detail.state).then(function () {
            WinJS.Application.sessionState.lastUrl = url;
        }));
    });

    WinJS.Namespace.define("SdkSample", {
        sampleTitle: sampleTitle,
        scenarios: scenarios,
        mediaCaptureMgr: null,
        photoFile: "photo.jpg",
        deviceList: null,
        recordState: null,
        captureInitSettings: null,
        encodingProfile: null,
        storageFile: null,
        photoStorage: null,
        cameraControlSliders: null,


        displayStatus: function (statusText) {
            WinJS.log && WinJS.log(statusText, "MediaCapture", "status");
        },

        displayError: function (error) {
            WinJS.log && WinJS.log(error, "MediaCapture", "error");
        },

        id: function (elementId) {
            return document.getElementById(elementId);
        },

    });

    WinJS.Application.addEventListener("activated", activated, false);
    WinJS.Application.start();
    Windows.UI.WebUI.WebUIApplication.addEventListener("suspending", SdkSample.suspendingHandler, false);
    Windows.UI.WebUI.WebUIApplication.addEventListener("resuming", SdkSample.resumingHandler, false);
})();
