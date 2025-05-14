//
//  MatQuickLook.mm
//
//  Created by Giles Payne on 2021/07/18.
//

#import "MatQuickLook.h"
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

@implementation MatQuickLook

+ (NSString*)makeLabel:(BOOL)isIntType val:(NSNumber*)num {
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

+ (void)relativeLine:(UIBezierPath*)path relX:(CGFloat)x relY:(CGFloat)y {
    CGPoint curr = path.currentPoint;
    [path addLineToPoint:CGPointMake(curr.x + x, curr.y + y)];
}

+ (id)matDebugQuickLookObject:(Mat*)mat {
    if ([mat dims] == 2 && [mat rows] <= 10 && [mat cols] <= 10 && [mat channels] == 1) {
        FontGetter fontGetters[] = { getCMU, getBodoni72, getAnySerif, getSystemFont };
        UIFont* font = nil;
        for (int fontGetterIndex = 0; font==nil && fontGetterIndex < (sizeof(fontGetters)) / (sizeof(fontGetters[0])); fontGetterIndex++) {
            font = fontGetters[fontGetterIndex]();
        }
        int elements = [mat rows] * [mat cols];
        NSDictionary<NSAttributedStringKey,id>* textFontAttributes = @{ NSFontAttributeName: font, NSForegroundColorAttributeName: UIColor.blackColor };
        NSMutableArray<NSNumber*>* rawData = [NSMutableArray new];
        for (int dataIndex = 0; dataIndex < elements; dataIndex++) {
            [rawData addObject:[NSNumber numberWithDouble:0]];
        }
        [mat get:0 col: 0 data: rawData];
        BOOL isIntType = [mat depth] <= CV_32S;
        NSMutableArray<NSString*>* labels = [NSMutableArray new];
        NSMutableDictionary<NSString*, NSValue*>* boundingRects = [NSMutableDictionary dictionaryWithCapacity:elements];
        int maxWidth = 0, maxHeight = 0;
        for (NSNumber* number in rawData) {
            NSString* label = [MatQuickLook makeLabel:isIntType val:number];
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
        int imageWidth = 2 * (borderGap + lipWidth) + maxWidth * [mat cols] + colGap * ([mat cols] - 1);
        int imageHeight = 2 * (borderGap + lipWidth) + maxHeight * [mat rows] + rowGap * ([mat rows] - 1);

        UIBezierPath* leftBracket = [UIBezierPath new];
        [leftBracket moveToPoint:CGPointMake(borderGap, borderGap)];
        [MatQuickLook relativeLine:leftBracket relX:0 relY:imageHeight - 2 * borderGap];
        [MatQuickLook relativeLine:leftBracket relX:lineThickness + lipWidth relY:0];
        [MatQuickLook relativeLine:leftBracket relX:0 relY:-lineThickness];
        [MatQuickLook relativeLine:leftBracket relX:-lipWidth relY:0];
        [MatQuickLook relativeLine:leftBracket relX:0 relY:-(imageHeight - 2 * (borderGap + lineThickness))];
        [MatQuickLook relativeLine:leftBracket relX:lipWidth relY:0];
        [MatQuickLook relativeLine:leftBracket relX:0 relY:-lineThickness];
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
            int row = (int)index / [mat cols];
            int col = (int)index % [mat cols];
            int x = borderGap + lipWidth + col * (maxWidth + colGap) + (maxWidth - boundingRect.size.width) / 2;
            int y = borderGap + lipWidth + row * (maxHeight + rowGap) + (maxHeight - boundingRect.size.height) / 2;
            CGRect textRect = CGRectMake(x, y, boundingRect.size.width, boundingRect.size.height);
            [label drawInRect:textRect withAttributes:textFontAttributes];
        }];
        UIImage* image = UIGraphicsGetImageFromCurrentImageContext();
        UIGraphicsEndImageContext();
        return image;
    } else if (([mat dims] == 2) && ([mat type] == CV_8U || [mat type] == CV_8UC3 || [mat type] == CV_8UC4)) {
        return [mat toUIImage];
    } else if ([mat dims] == 2 && [mat channels] == 1) {
        Mat* normalized = [Mat new];
        [Core normalize:mat dst:normalized alpha:0 beta:255 norm_type:NORM_MINMAX dtype:CV_8U];
        Mat* normalizedKey = [[Mat alloc] initWithRows:[mat rows] + 10 cols:[mat cols] type:CV_8U];
        std::vector<char> key;
        for (int index = 0; index < [mat cols]; index++) {
            key.push_back((char)(index * 256 / [mat cols]));
        }
        for (int index = 0; index < 10; index++) {
            [normalizedKey put:@[[NSNumber numberWithInt:index], [NSNumber numberWithInt:0]] count:[mat cols] byteBuffer:key.data()];
        }
        [normalized copyTo:[normalizedKey submatRoi:[[Rect2i alloc] initWithX:0 y:10 width:[mat cols] height:[mat rows]]]];
        Mat* colorMap = [Mat new];
        [Imgproc applyColorMap:normalizedKey dst:colorMap colormap:COLORMAP_JET];
        [Imgproc cvtColor:colorMap dst:colorMap code:COLOR_BGR2RGB];
        return [colorMap toUIImage];
    }
    return [mat description];
}

@end
