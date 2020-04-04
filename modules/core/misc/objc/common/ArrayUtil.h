//
//  ArrayUtil.h
//
//  Created by Giles Payne on 2020/02/09.
//

#pragma once

@interface NSMutableArray (Autosizing)
+(NSMutableArray*)allocateWithSize:(NSInteger)size fillValue:(NSObject*)val;
@end
