//
//  Mat+QuickLook.mm
//
//  Created by Giles Payne on 2021/07/18.
//

#import "Mat+QuickLook.h"
#import "Mat+Converters.h"
#import "Rect2i.h"
#import "Core.h"
#import "Imgproc.h"
#import <opencv2/imgcodecs/macosx.h>

#define SIZE 20

static NSFont* getCMU() {
    return [NSFont fontWithName:@"CMU Serif" size:SIZE];
}

static NSFont* getBodoni72() {
    return [NSFont fontWithName:@"Bodoni 72" size:SIZE];
}

static NSFont* getAnySerif() {
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 110000
    if (@available(macOS 11.0, *)) {
        return [NSFont fontWithDescriptor:[[NSFontDescriptor preferredFontDescriptorForTextStyle:NSFontTextStyleBody options:@{}] fontDescriptorWithDesign:NSFontDescriptorSystemDesignSerif] size:SIZE];
    } else {
        return nil;
    }
#else
    return nil;
#endif
}

static NSFont* getSystemFont() {
    return [NSFont systemFontOfSize:SIZE];
}

typedef NSFont* (*FontGetter)();

@implementation Mat (QuickLook)

- (NSString*)makeLabel:(BOOL)isIntType val:(NSNumber*)num {
    if (isIntType) {
        return [NSString stringWithFormat:@"%d", num.intValue];
    } else {
        int exponent = 1 + (int)log10(abs(num.doubleValue));
        if (num.doubleValue == (double)num.intValue && num.doubleValue < 10000 && num.doubleValue > -10000) {
            return [NSString stringWithFormat:@"%d", num.intValue];;
        } else if (exponent <= 5 && exponent >= -1) {
            return [NSString stringWithFormat:[NSString stringWithFormat:@"%%%d.%df", 6, MIN(5 - exponent, 4)], num.doubleValue];
        } else {
            return [[[NSString stringWithFormat:@"%.2e", num.doubleValue] stringByReplacingOccurrencesOfString:@"e+0" withString:@"e"] stringByReplacingOccurrencesOfString:@"e-0" withString:@"e-"];
        }
    }
}

- (id)debugQuickLookObject {
    // for smallish Mat objects display as a matrix
    if ([self dims] == 2 && [self rows] <= 10 && [self cols] <= 10 && [self channels] == 1) {
        FontGetter fontGetters[] = { getCMU, getBodoni72, getAnySerif, getSystemFont };
        NSFont* font = nil;
        for (int fontGetterIndex = 0; font==nil && fontGetterIndex < (sizeof(fontGetters)) / (sizeof(fontGetters[0])); fontGetterIndex++) {
            font = fontGetters[fontGetterIndex]();
        }
        int elements = [self rows] * [self cols];
        NSDictionary<NSAttributedStringKey,id>* textFontAttributes = @{ NSFontAttributeName: font, NSForegroundColorAttributeName: NSColor.blackColor };
        NSMutableArray<NSNumber*>* rawData = [NSMutableArray new];
        for (int dataIndex = 0; dataIndex < elements; dataIndex++) {
            [rawData addObject:[NSNumber numberWithDouble:0]];
        }
        [self get:0 col: 0 data: rawData];
        BOOL isIntType = [self depth] <= CV_32S;
        NSMutableArray<NSString*>* labels = [NSMutableArray new];
        NSMutableDictionary<NSString*, NSValue*>* boundingRects = [NSMutableDictionary dictionaryWithCapacity:elements];
        int maxWidth = 0, maxHeight = 0;
        for (NSNumber* number in rawData) {
            NSString* label = [self makeLabel:isIntType val:number];
            [labels addObject:label];
            NSRect boundingRect = [label boundingRectWithSize:NSMakeSize(CGFLOAT_MAX, CGFLOAT_MAX) options:NSStringDrawingUsesLineFragmentOrigin attributes:textFontAttributes];
            if (boundingRect.size.width > maxWidth) {
                maxWidth = boundingRect.size.width;
            }
            if (boundingRect.size.height > maxHeight) {
                maxHeight = boundingRect.size.height;
            }
            boundingRects[label] = [NSValue valueWithRect:boundingRect];
        }

        int rowGap = 8;
        int colGap = 8;
        int borderGap = 9;
        int lineThickness = 4;
        int lipWidth = 8;
        int imageWidth = 2 * (borderGap + lipWidth) + maxWidth * [self cols] + colGap * ([self cols] - 1);
        int imageHeight = 2 * (borderGap + lipWidth) + maxHeight * [self rows] + rowGap * ([self rows] - 1);
        NSImage* image = [[NSImage alloc] initWithSize:NSMakeSize(imageWidth, imageHeight)];
        NSBezierPath* leftBracket = [NSBezierPath new];
        [leftBracket moveToPoint:NSMakePoint(borderGap, borderGap)];
        [leftBracket relativeLineToPoint:NSMakePoint(0, imageHeight - 2 * borderGap)];
        [leftBracket relativeLineToPoint:NSMakePoint(lineThickness + lipWidth, 0)];
        [leftBracket relativeLineToPoint:NSMakePoint(0, -lineThickness)];
        [leftBracket relativeLineToPoint:NSMakePoint(-lipWidth, 0)];
        [leftBracket relativeLineToPoint:NSMakePoint(0, -(imageHeight - 2 * (borderGap + lineThickness)))];
        [leftBracket relativeLineToPoint:NSMakePoint(lipWidth, 0)];
        [leftBracket relativeLineToPoint:NSMakePoint(0, -lineThickness)];
        [leftBracket relativeLineToPoint:NSMakePoint(-(lineThickness + lipWidth), 0)];
        NSAffineTransform* reflect = [NSAffineTransform new];
        [reflect scaleXBy:-1 yBy:1];
        [reflect translateXBy:-imageWidth yBy:0];
        NSBezierPath* rightBracket = [leftBracket copy];
        [rightBracket transformUsingAffineTransform:reflect];

        [image lockFocus];
        [NSColor.whiteColor drawSwatchInRect:NSMakeRect(0, 0, imageWidth, imageHeight)];
        [NSColor.blackColor set];
        [leftBracket fill];
        [rightBracket fill];

        [labels enumerateObjectsUsingBlock:^(id label, NSUInteger index, BOOL *stop)
        {
            NSRect boundingRect = boundingRects[label].rectValue;
            int row = [self rows] - 1 - ((int)index / [self cols]);
            int col = (int)index % [self cols];
            int x = borderGap + lipWidth + col * (maxWidth + colGap) + (maxWidth - boundingRect.size.width) / 2;
            int y = borderGap + lipWidth + row * (maxHeight + rowGap) + (maxHeight - boundingRect.size.height) / 2;
            NSRect textRect = NSMakeRect(x, y, boundingRect.size.width, boundingRect.size.height);
            [label drawInRect:textRect withAttributes:textFontAttributes];
        }];
        [image unlockFocus];
        return image;
    } else if (([self dims] == 2) && ([self type] == CV_8U || [self type] == CV_8UC3 || [self type] == CV_8UC4)) {
        // convert to NSImage if the Mats has 2 dimensions and a type and number of channels consistent with it being a image
        return [self toNSImage];
    } else if ([self dims] == 2 && [self channels] == 1) {
        // for other Mats with 2 dimensions and one channel - generate heat map
        Mat* normalized = [Mat new];
        [Core normalize:self dst:normalized alpha:0 beta:255 norm_type:NORM_MINMAX dtype:CV_8U];
        Mat* normalizedKey = [[Mat alloc] initWithRows:[self rows] + 10 cols:[self cols] type:CV_8U];
        std::vector<char> key;
        for (int index = 0; index < [self cols]; index++) {
            key.push_back((char)(index * 256 / [self cols]));
        }
        for (int index = 0; index < 10; index++) {
            [normalizedKey put:@[[NSNumber numberWithInt:index], [NSNumber numberWithInt:0]] count:[self cols] byteBuffer:key.data()];
        }
        [normalized copyTo:[normalizedKey submatRoi:[[Rect2i alloc] initWithX:0 y:10 width:[self cols] height:[self rows]]]];
        Mat* colorMap = [Mat new];
        [Imgproc applyColorMap:normalizedKey dst:colorMap colormap:COLORMAP_JET];
        [Imgproc cvtColor:colorMap dst:colorMap code:COLOR_BGR2RGB];
        return [colorMap toNSImage];
    }
    //everything just return the Mat description
    return [self description];
}

@end
