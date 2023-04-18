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
#import <opencv2/imgcodecs/ios.h>

#define SIZE 20

static UIFont* getCMU() {
    return [UIFont fontWithName:@"CMU Serif" size:SIZE];
}

static UIFont* getBodoni72() {
    return [UIFont fontWithName:@"Bodoni 72" size:SIZE];
}

static UIFont* getAnySerif() {
#if defined(__IPHONE_OS_VERSION_MAX_ALLOWED) && __IPHONE_OS_VERSION_MAX_ALLOWED >= 130000
    if (@available(iOS 13.0, *)) {
        return [UIFont fontWithDescriptor:[[UIFontDescriptor preferredFontDescriptorWithTextStyle:UIFontTextStyleBody] fontDescriptorWithDesign:UIFontDescriptorSystemDesignSerif] size:SIZE];
    } else {
        return nil;
    }
#else
    return nil;
#endif
}

static UIFont* getSystemFont() {
    return [UIFont systemFontOfSize:SIZE];
}

typedef UIFont* (*FontGetter)();

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

- (void)relativeLine:(UIBezierPath*)path relX:(CGFloat)x relY:(CGFloat)y {
    CGPoint curr = path.currentPoint;
    [path addLineToPoint:CGPointMake(curr.x + x, curr.y + y)];
}

- (id)debugQuickLookObject {
    if ([self dims] == 2 && [self rows] <= 10 && [self cols] <= 10 && [self channels] == 1) {
        FontGetter fontGetters[] = { getCMU, getBodoni72, getAnySerif, getSystemFont };
        UIFont* font = nil;
        for (int fontGetterIndex = 0; font==nil && fontGetterIndex < (sizeof(fontGetters)) / (sizeof(fontGetters[0])); fontGetterIndex++) {
            font = fontGetters[fontGetterIndex]();
        }
        int elements = [self rows] * [self cols];
        NSDictionary<NSAttributedStringKey,id>* textFontAttributes = @{ NSFontAttributeName: font, NSForegroundColorAttributeName: UIColor.blackColor };
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
            CGRect boundingRect = [label boundingRectWithSize:CGSizeMake(CGFLOAT_MAX, CGFLOAT_MAX) options:NSStringDrawingUsesLineFragmentOrigin attributes:textFontAttributes context:nil];
            if (boundingRect.size.width > maxWidth) {
                maxWidth = boundingRect.size.width;
            }
            if (boundingRect.size.height > maxHeight) {
                maxHeight = boundingRect.size.height;
            }
            boundingRects[label] = [NSValue valueWithCGRect:boundingRect];
        }

        int rowGap = 6;
        int colGap = 6;
        int borderGap = 8;
        int lineThickness = 3;
        int lipWidth = 6;
        int imageWidth = 2 * (borderGap + lipWidth) + maxWidth * [self cols] + colGap * ([self cols] - 1);
        int imageHeight = 2 * (borderGap + lipWidth) + maxHeight * [self rows] + rowGap * ([self rows] - 1);

        UIBezierPath* leftBracket = [UIBezierPath new];
        [leftBracket moveToPoint:CGPointMake(borderGap, borderGap)];
        [self relativeLine:leftBracket relX:0 relY:imageHeight - 2 * borderGap];
        [self relativeLine:leftBracket relX:lineThickness + lipWidth relY:0];
        [self relativeLine:leftBracket relX:0 relY:-lineThickness];
        [self relativeLine:leftBracket relX:-lipWidth relY:0];
        [self relativeLine:leftBracket relX:0 relY:-(imageHeight - 2 * (borderGap + lineThickness))];
        [self relativeLine:leftBracket relX:lipWidth relY:0];
        [self relativeLine:leftBracket relX:0 relY:-lineThickness];
        [leftBracket closePath];
        CGAffineTransform reflect = CGAffineTransformConcat(CGAffineTransformMakeTranslation(-imageWidth, 0), CGAffineTransformMakeScale(-1, 1));
        UIBezierPath* rightBracket = [leftBracket copy];
        [rightBracket applyTransform:reflect];

        CGRect rect = CGRectMake(0, 0, imageWidth, imageHeight);
        UIGraphicsBeginImageContextWithOptions(rect.size, false, 0.0);
        [UIColor.whiteColor setFill];
        UIRectFill(rect);
        [UIColor.blackColor setFill];
        [leftBracket fill];
        [rightBracket fill];
        [labels enumerateObjectsUsingBlock:^(id label, NSUInteger index, BOOL *stop)
        {
            CGRect boundingRect = boundingRects[label].CGRectValue;
            int row = (int)index / [self cols];
            int col = (int)index % [self cols];
            int x = borderGap + lipWidth + col * (maxWidth + colGap) + (maxWidth - boundingRect.size.width) / 2;
            int y = borderGap + lipWidth + row * (maxHeight + rowGap) + (maxHeight - boundingRect.size.height) / 2;
            CGRect textRect = CGRectMake(x, y, boundingRect.size.width, boundingRect.size.height);
            [label drawInRect:textRect withAttributes:textFontAttributes];
        }];
        UIImage* image = UIGraphicsGetImageFromCurrentImageContext();
        UIGraphicsEndImageContext();
        return image;
    } else if (([self dims] == 2) && ([self type] == CV_8U || [self type] == CV_8UC3 || [self type] == CV_8UC4)) {
        return [self toUIImage];
    } else if ([self dims] == 2 && [self channels] == 1) {
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
        return [colorMap toUIImage];
    }
    return [self description];
}

@end
