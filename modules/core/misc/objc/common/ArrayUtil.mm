//
//  ArrayUtil.mm
//
//  Created by Giles Payne on 2020/02/09.
//

#import "ArrayUtil.h"

NSMutableArray* createArrayWithSize(int size, NSObject* val) {
    NSMutableArray *array = [NSMutableArray arrayWithCapacity:size];
    for (int i = 0; i < size; i++){
        [array addObject:val];
    }
    return array;
}
