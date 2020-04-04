//
//  IntOut.m
//
//  Created by Giles Payne on 2020/02/05.
//

#import "IntOut.h"

@implementation IntOut {
    int _val;
}

-(int)val {
    return _val;
}

-(int*)ptr {
    return &_val;
}

@end
