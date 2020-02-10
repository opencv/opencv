//
//  ArrayUtils.m
//
//  Created by Giles Payne on 2020/02/09.
//

#import <Foundation/Foundation.h>
#import "ArrayUtil.h"

@implementation NSMutableArray (Autosizing)
+(NSMutableArray*)allocateWithSize:(NSInteger)size fillValue:(NSObject *)val
{
    NSMutableArray *array = [NSMutableArray arrayWithCapacity:size];
    for (int i = 0; i < size; i++){
        [array addObject:val];
    }
    return array;
}
@end
