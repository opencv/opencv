// Based on http://www.phoboslab.org/log/2009/07/uvc-camera-control-for-mac-os-x

#import "UVCCameraControl.h"


const uvc_controls_t uvc_controls = {
    // Camera Terminal Control Selectors
    .autoExposure = {
        .unit = UVC_INPUT_TERMINAL_ID,
        .selector = 0x02, // CT_AE_MODE_CONTROL
        .size = 1,
    },
    .exposure = {
        .unit = UVC_INPUT_TERMINAL_ID,
        .selector = 0x04, // CT_EXPOSURE_TIME_ABSOLUTE_CONTROL
        .size = 4,
    },
    .focus = {
        .unit = UVC_INPUT_TERMINAL_ID,
        .selector = 0x06, // CT_FOCUS_ABSOLUTE_CONTROL
        .size = 2,
    },
    .autoFocus = {
        .unit = UVC_INPUT_TERMINAL_ID,
        .selector = 0x08, // CT_FOCUS_AUTO_CONTROL
        .size = 1,
    },
    .zoom = {
        .unit = UVC_INPUT_TERMINAL_ID,
        .selector = 0x0B, // CT_ZOOM_ABSOLUTE_CONTROL
        .size = 2,
    },

    // Processing Unit Control Selectors
    .brightness = {
        .unit = UVC_PROCESSING_UNIT_ID,
        .selector = 0x02,
        .size = 2,
    },
    .contrast = {
        .unit = UVC_PROCESSING_UNIT_ID,
        .selector = 0x03,
        .size = 2,
    },
    .gain = {
        .unit = UVC_PROCESSING_UNIT_ID,
        .selector = 0x04,
        .size = 2,
    },
    .saturation = {
        .unit = UVC_PROCESSING_UNIT_ID,
        .selector = 0x07,
        .size = 2,
    },
    .sharpness = {
        .unit = UVC_PROCESSING_UNIT_ID,
        .selector = 0x08,
        .size = 2,
    },
    .whiteBalance = {
        .unit = UVC_PROCESSING_UNIT_ID,
        .selector = 0x0A, // PU_WHITE_BALANCE_TEMPERA TURE_CONTROL
        .size = 2,
    },
    .autoWhiteBalance = {
        .unit = UVC_PROCESSING_UNIT_ID,
        .selector = 0x0B, // PU_WHITE_BALANCE_TEMPERATURE_AUTO_CONTROL
        .size = 1,
    },
};


@implementation UVCCameraControl

- (id)initWithLocationID:(UInt32)locationID {
    if( self = [super init] ) {
        interface = NULL;

        // Find All USB Devices, get their locationId and check if it matches the requested one
        CFMutableDictionaryRef matchingDict = IOServiceMatching(kIOUSBDeviceClassName);
        io_iterator_t serviceIterator;
        IOServiceGetMatchingServices( kIOMasterPortDefault, matchingDict, &serviceIterator );

        io_service_t camera;
        while( (camera = IOIteratorNext(serviceIterator)) ) {
            // Get DeviceInterface
            IOUSBDeviceInterface **deviceInterface = NULL;
            IOCFPlugInInterface **plugInInterface = NULL;
            SInt32 score;
            kern_return_t kr = IOCreatePlugInInterfaceForService( camera, kIOUSBDeviceUserClientTypeID, kIOCFPlugInInterfaceID, &plugInInterface, &score );
            if( (kIOReturnSuccess != kr) || !plugInInterface ) {
                NSLog( @"CameraControl Error: IOCreatePlugInInterfaceForService returned 0x%08x.", kr );
                continue;
            }

            HRESULT res = (*plugInInterface)->QueryInterface(plugInInterface, CFUUIDGetUUIDBytes(kIOUSBDeviceInterfaceID), (LPVOID*) &deviceInterface );
            (*plugInInterface)->Release(plugInInterface);
            if( res || deviceInterface == NULL ) {
                NSLog( @"CameraControl Error: QueryInterface returned %d.\n", (int)res );
                continue;
            }

            UInt32 currentLocationID = 0;
            (*deviceInterface)->GetLocationID(deviceInterface, &currentLocationID);

            if( currentLocationID == locationID ) {
                // Yep, this is the USB Device that was requested!
                interface = [self getControlInferaceWithDeviceInterface:deviceInterface];
                [self updateCapabilities];
                return self;
            }
        } // end while

    }

    return self;
}

- (id)initWithVendorID:(long)vendorID productID:(long)productID {
    if( self = [super init] ) {
        interface = NULL;

        // Find USB Device
        CFMutableDictionaryRef matchingDict = IOServiceMatching(kIOUSBDeviceClassName);
        CFNumberRef numberRef;

        numberRef = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &vendorID);
        CFDictionarySetValue( matchingDict, CFSTR(kUSBVendorID), numberRef );
        CFRelease(numberRef);

        numberRef = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &productID);
        CFDictionarySetValue( matchingDict, CFSTR(kUSBProductID), numberRef );
        CFRelease(numberRef);
        io_service_t camera = IOServiceGetMatchingService( kIOMasterPortDefault, matchingDict );


        // Get DeviceInterface
        IOUSBDeviceInterface **deviceInterface = NULL;
        IOCFPlugInInterface **plugInInterface = NULL;
        SInt32 score;
        kern_return_t kr = IOCreatePlugInInterfaceForService( camera, kIOUSBDeviceUserClientTypeID, kIOCFPlugInInterfaceID, &plugInInterface, &score );
        if( (kIOReturnSuccess != kr) || !plugInInterface ) {
            NSLog( @"CameraControl Error: IOCreatePlugInInterfaceForService returned 0x%08x.", kr );
            return self;
        }

        HRESULT res = (*plugInInterface)->QueryInterface(plugInInterface, CFUUIDGetUUIDBytes(kIOUSBDeviceInterfaceID), (LPVOID*) &deviceInterface );
        (*plugInInterface)->Release(plugInInterface);
        if( res || deviceInterface == NULL ) {
            NSLog( @"CameraControl Error: QueryInterface returned %d.\n", (int)res );
            return self;
        }

        interface = [self getControlInferaceWithDeviceInterface:deviceInterface];
        [self updateCapabilities];
    }
    return self;
}

- (IOUSBInterfaceInterface190 **)getControlInferaceWithDeviceInterface:(IOUSBDeviceInterface **)deviceInterface {
    IOUSBInterfaceInterface190 **controlInterface;

    io_iterator_t interfaceIterator;
    IOUSBFindInterfaceRequest interfaceRequest;
    interfaceRequest.bInterfaceClass = UVC_CONTROL_INTERFACE_CLASS;
    interfaceRequest.bInterfaceSubClass = UVC_CONTROL_INTERFACE_SUBCLASS;
    interfaceRequest.bInterfaceProtocol = kIOUSBFindInterfaceDontCare;
    interfaceRequest.bAlternateSetting = kIOUSBFindInterfaceDontCare;

    IOReturn success = (*deviceInterface)->CreateInterfaceIterator( deviceInterface, &interfaceRequest, &interfaceIterator );
    if( success != kIOReturnSuccess ) {
        return NULL;
    }

    io_service_t usbInterface;
    HRESULT result;


    if( (usbInterface = IOIteratorNext(interfaceIterator)) ) {
        IOCFPlugInInterface **plugInInterface = NULL;

        //Create an intermediate plug-in
        SInt32 score;
        kern_return_t kr = IOCreatePlugInInterfaceForService( usbInterface, kIOUSBInterfaceUserClientTypeID, kIOCFPlugInInterfaceID, &plugInInterface, &score );

        //Release the usbInterface object after getting the plug-in
        kr = IOObjectRelease(usbInterface);
        if( (kr != kIOReturnSuccess) || !plugInInterface ) {
            NSLog( @"CameraControl Error: Unable to create a plug-in (%08x)\n", kr );
            return NULL;
        }

        //Now create the device interface for the interface
        result = (*plugInInterface)->QueryInterface( plugInInterface, CFUUIDGetUUIDBytes(kIOUSBInterfaceInterfaceID), (LPVOID *) &controlInterface );

        //No longer need the intermediate plug-in
        (*plugInInterface)->Release(plugInInterface);

        if( result || !controlInterface ) {
            NSLog( @"CameraControl Error: Couldn’t create a device interface for the interface (%08x)", (int) result );
            return NULL;
        }

        return controlInterface;
    }

    return NULL;
}

- (void)dealloc {
    if( interface ) {
        (*interface)->USBInterfaceClose(interface);
        (*interface)->Release(interface);
    }
    [super dealloc];
}

- (BOOL)sendControlRequest:(IOUSBDevRequest)controlRequest {
    if( !interface ){
        NSLog( @"CameraControl Error: No interface to send request" );
        return NO;
    }

    //Now open the interface. This will cause the pipes associated with
    //the endpoints in the interface descriptor to be instantiated
    kern_return_t kr = (*interface)->USBInterfaceOpen(interface);
    if (kr != kIOReturnSuccess) {
        NSLog( @"CameraControl Error: Unable to open interface (%08x)\n", kr );
        return NO;
    }

    kr = (*interface)->ControlRequest( interface, 0, &controlRequest );
    if( kr != kIOReturnSuccess ) {
        kr = (*interface)->USBInterfaceClose(interface);
        // NSLog( @"CameraControl Error: Control request failed: %08x", kr);
        return NO;
    }

    kr = (*interface)->USBInterfaceClose(interface);

    return YES;
}

- (BOOL)setData:(long)value withLength:(int)length forSelector:(int)selector at:(int)unitId {
    IOUSBDevRequest controlRequest;
    controlRequest.bmRequestType = USBmakebmRequestType( kUSBOut, kUSBClass, kUSBInterface );
    controlRequest.bRequest = UVC_SET_CUR;
    controlRequest.wValue = (selector << 8) | 0x00;
    controlRequest.wIndex = (unitId << 8) | 0x00;
    controlRequest.wLength = length;
    controlRequest.wLenDone = 0;
    controlRequest.pData = &value;
    BOOL success = [self sendControlRequest:controlRequest];
    if (!success) {
        NSLog(@"CameraControl: request 0x%2x failed: selector: 0x%02x, length: %d, value: %ld", UVC_SET_CUR, selector, length, value);
    }
    return success;
}

- (long)getDataFor:(int)type withLength:(int)length fromSelector:(int)selector at:(int)unitId {
    long value = 0;
    IOUSBDevRequest controlRequest;
    controlRequest.bmRequestType = USBmakebmRequestType( kUSBIn, kUSBClass, kUSBInterface );
    controlRequest.bRequest = type;
    controlRequest.wValue = (selector << 8) | 0x00;
    controlRequest.wIndex = (unitId << 8) | 0x00;
    controlRequest.wLength = length;
    controlRequest.wLenDone = 0;
    controlRequest.pData = &value;
    BOOL success = [self sendControlRequest:controlRequest];
    if (!success) {
        NSLog(@"CameraControl: request 0x%2x failed: selector: 0x%02x, length: %d, value: %ld", type, selector, length, value);
    }
    return ( success ? value : -1 );
}

// Get Range (min, max)
- (uvc_range_t)getRangeForControl:(const uvc_control_info_t *)control {
    uvc_range_t range = { 0, 0 };
    range.min = [self getDataFor:UVC_GET_MIN withLength:control->size fromSelector:control->selector at:control->unit];
    range.max = [self getDataFor:UVC_GET_MAX withLength:control->size fromSelector:control->selector at:control->unit];
    return range;
}


// Used to de-/normalize values
- (double)mapValue:(double)value fromMin:(double)fromMin max:(double)fromMax toMin:(double)toMin max:(double)toMax {
    return toMin + (toMax - toMin) * ((value - fromMin) / (fromMax - fromMin));
}

- (long)getInfoForControl:(const uvc_control_info_t *)control {
    return [self getDataFor:UVC_GET_INFO
                 withLength:1
               fromSelector:control->selector
                         at:control->unit];
}

- (long)getResolutionForControl:(const uvc_control_info_t *)control {
    return [self getDataFor:UVC_GET_RES
                 withLength:control->size
               fromSelector:control->selector
                         at:control->unit];
}

- (long)getValueForControl:(const uvc_control_info_t *)control {
    return [self getDataFor:UVC_GET_CUR
                 withLength:control->size
               fromSelector:control->selector
                         at:control->unit];
}

- (long)getDefaultValueForControl:(const uvc_control_info_t *)control {
    return [self getDataFor:UVC_GET_DEF
                 withLength:control->size
               fromSelector:control->selector
                         at:control->unit];
}

- (BOOL)setValue:(long)value forControl:(const uvc_control_info_t *)control {
    return [self setData:value
              withLength:control->size
             forSelector:control->selector
                      at:control->unit];
}

- (void)updateCapabilities {
    supportedAutoExposure = [self getResolutionForControl:&uvc_controls.autoExposure];
    focusRange = [self getRangeForControl:&uvc_controls.focus];
    exposureRange = [self getRangeForControl:&uvc_controls.exposure];
}


// === Exposure ================================================================

// From USB Video Class spec:
#define UVC_AUTO_EXPOSURE_MANUAL    0x01 // Manual Mode – manual Exposure Time, manual Iris
#define UVC_AUTO_EXPOSURE_AUTO      0x02 // Auto Mode – auto Exposure Time, auto Iris
#define UVC_AUTO_EXPOSURE_SHUTTER   0x04 // Shutter Priority Mode – manual Exposure Time, auto Iris
#define UVC_AUTO_EXPOSURE_APERTURE  0x08 // Aperture Priority Mode – auto Exposure Time, manual Iris

- (BOOL)setAutoExposure:(BOOL)enabled {
    // Most USB cameras don't support fully automatic mode; only Manual and
    // Aperture Priority. So we check that auto mode actually works and fall back
    // to one of the other modes.
    long value = UVC_AUTO_EXPOSURE_MANUAL;
    if (enabled) {
        if (supportedAutoExposure & UVC_AUTO_EXPOSURE_AUTO) value = UVC_AUTO_EXPOSURE_AUTO;
        else if (supportedAutoExposure & UVC_AUTO_EXPOSURE_APERTURE) value = UVC_AUTO_EXPOSURE_APERTURE;
        else if (supportedAutoExposure & UVC_AUTO_EXPOSURE_SHUTTER) value = UVC_AUTO_EXPOSURE_SHUTTER;
        else return NO;
    }
    return [self setValue:value forControl:&uvc_controls.autoExposure];
}

- (BOOL)getAutoExposure {
    long value = [self getValueForControl:&uvc_controls.autoExposure];
    return (value >= UVC_AUTO_EXPOSURE_AUTO ? YES : NO);
}

- (BOOL)setExposure:(double)value {
    if (![self getAutoExposure]) {
        return [self setValue:value forControl:&uvc_controls.exposure];
    } else {
        return NO;
    }
}

- (double)getExposure {
    return [self getValueForControl:&uvc_controls.exposure];
}

// === Focus ===================================================================

- (BOOL)setAutoFocus:(BOOL)value {
    return [self setValue:(value ? 0x01 : 0x00) forControl:&uvc_controls.autoFocus];
}

- (BOOL)getAutoFocus {
    long value = [self getValueForControl:&uvc_controls.autoFocus];
    return (value ? YES : NO);
}

- (BOOL)setFocus:(double)value {
    // Don't try to set the focus if autofocus is enabled; it'll fail anyway.
    if (![self getAutoFocus]) {
        long focus = [self mapValue:value fromMin:0 max:1 toMin:focusRange.min max:focusRange.max];
        return [self setValue:focus forControl:&uvc_controls.focus];
    } else {
        return NO;
    }
}

- (double)getFocus {
    long focus = [self getValueForControl:&uvc_controls.focus];
    return [self mapValue:focus fromMin:focusRange.min max:focusRange.max toMin:0 max:1];
}


@end
