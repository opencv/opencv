//
//  FloatOut.h
//
//  Created by Giles Payne on 2020/02/05.
//

#pragma once

#import <Foundation/Foundation.h>

/**
* Simple wrapper class to handle out parameters of type `float`
*/
@interface FloatOut : NSObject

/**
* The `float` value
*/
@property(readonly) float val;

/**
* Pointer to the `float` value
*/
-(float*)ptr;

@end
