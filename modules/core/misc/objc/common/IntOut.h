//
//  IntOut.h
//
//  Created by Giles Payne on 2020/02/05.
//

#pragma once

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
* Simple wrapper class to handle out parameters of type `int`
*/
@interface IntOut : NSObject

/**
* The `int` value
*/
@property(readonly) int val;

/**
* Pointer to the `int` value
*/
-(int*)ptr;

@end

NS_ASSUME_NONNULL_END
