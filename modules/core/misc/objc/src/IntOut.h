//
//  IntOut.h
//
//  Created by Giles Payne on 2020/02/05.
//

#pragma once

#import <Foundation/Foundation.h>

@interface IntOut : NSObject

@property(readonly) int val;

-(int*)ptr;

@end
