//
//  ArrayUtil.h
//
//  Created by Giles Payne on 2020/02/09.
//

#pragma once

#import <Foundation/Foundation.h>

/**
* Utility function to create and populate an NSMutableArray with a specific size
* @param size Size of array to create
* @param val Value with which to initialize array elements
*/
NSMutableArray* createArrayWithSize(int size, NSObject* val);
