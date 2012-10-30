#import <Foundation/Foundation.h>
#include <CoreFoundation/CoreFoundation.h>

#include <IOKit/IOKitLib.h>
#include <IOKit/IOMessage.h>
#include <IOKit/IOCFPlugIn.h>
#include <IOKit/usb/IOUSBLib.h>


#define UVC_INPUT_TERMINAL_ID 	0x01
#define UVC_PROCESSING_UNIT_ID 	0x02

#define UVC_CONTROL_INTERFACE_CLASS 	14
#define UVC_CONTROL_INTERFACE_SUBCLASS 	1

#define UVC_SET_CUR		0x01
#define UVC_GET_CUR		0x81
#define UVC_GET_MIN		0x82
#define UVC_GET_MAX		0x83
#define UVC_GET_RES		0x84
#define UVC_GET_LEN		0x85
#define UVC_GET_INFO 	0x86
#define UVC_GET_DEF 	0x87

typedef struct {
	int min, max;
} uvc_range_t;

typedef struct {
	int unit;
	int selector;
	int size;
} uvc_control_info_t;

typedef struct {
	uvc_control_info_t autoExposure;
	uvc_control_info_t exposure;
	uvc_control_info_t focus;
	uvc_control_info_t autoFocus;
	uvc_control_info_t brightness;
	uvc_control_info_t contrast;
	uvc_control_info_t gain;
	uvc_control_info_t saturation;
	uvc_control_info_t sharpness;
	uvc_control_info_t whiteBalance;
	uvc_control_info_t autoWhiteBalance;
} uvc_controls_t ;


@interface UVCCameraControl : NSObject {
	long dataBuffer;
	IOUSBInterfaceInterface190 **interface;

	long supportedAutoExposure;
	uvc_range_t focusRange;
}


- (id)initWithLocationID:(UInt32)locationID;
- (id)initWithVendorID:(long)vendorID productID:(long)productID;
- (IOUSBInterfaceInterface190 **)getControlInferaceWithDeviceInterface:(IOUSBDeviceInterface **)deviceInterface;

- (BOOL)sendControlRequest:(IOUSBDevRequest)controlRequest;
- (BOOL)setData:(long)value withLength:(int)length forSelector:(int)selector at:(int)unitID;
- (long)getDataFor:(int)type withLength:(int)length fromSelector:(int)selector at:(int)unitID;
- (uvc_range_t)getRangeForControl:(const uvc_control_info_t *)control;
- (double)mapValue:(double)value fromMin:(double)fromMin max:(double)fromMax toMin:(double)toMin max:(double)toMax;
- (long)getInfoForControl:(const uvc_control_info_t *)control;
- (long)getResolutionForControl:(const uvc_control_info_t *)control;
- (long)getDefaultValueForControl:(const uvc_control_info_t *)control;
- (long)getValueForControl:(const uvc_control_info_t *)control;
- (BOOL)setValue:(long)value forControl:(const uvc_control_info_t *)control;
- (void)updateCapabilities;

- (BOOL)setAutoExposure:(BOOL)enabled;
- (BOOL)getAutoExposure;
- (BOOL)setExposure:(double)value;
- (double)getExposure;

- (BOOL)setAutoFocus:(BOOL)enabled;
- (BOOL)getAutoFocus;
- (BOOL)setFocus:(double)value;
- (double)getFocus;

// - (BOOL)setGain:(double)value;
// - (double)getGain;
// - (BOOL)setBrightness:(double)value;
// - (double)getBrightness;
// - (BOOL)setContrast:(double)value;
// - (double)getContrast;
// - (BOOL)setSaturation:(double)value;
// - (double)getSaturation;
// - (BOOL)setSharpness:(double)value;
// - (double)getSharpness;

@end
