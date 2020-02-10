//
//  IntOut.m
//
//  Created by Giles Payne on 2020/02/05.
//

#import "DoubleOut.h"

@implementation DoubleOut {
    double _val;
}

-(double)val {
    return _val;
}

-(double*)ptr {
    return &_val;
}

@end
