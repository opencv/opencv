//
//  IntOut.m
//
//  Created by Giles Payne on 2020/02/05.
//

#import "FloatOut.h"

@implementation FloatOut {
    float _val;
}

-(float)val {
    return _val;
}

-(float*)ptr {
    return &_val;
}

@end
