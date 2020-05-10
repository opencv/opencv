//
//  DoubleOut.h
//
//  Created by Giles Payne on 2020/02/05.
//

#pragma once

#import <Foundation/Foundation.h>

/**
* Simple wrapper class to handle out parameters of type `double`
*/
@interface DoubleOut : NSObject

#pragma mark - Property

/**
* The `double` value
*/
@property(readonly) double val;

/**
* Pointer to the `double` value
*/
-(double*)ptr;

@end
