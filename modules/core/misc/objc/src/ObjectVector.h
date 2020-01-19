//
//  IntVector.h
//
//  Created by Giles Payne on 2020/01/04.
//

#pragma once

#import <Foundation/Foundation.h>
#ifdef __cplusplus
#import <vector>
#endif

NS_ASSUME_NONNULL_BEGIN
@interface ObjectVector<ObjectType> : NSObject<NSFastEnumeration>

-(instancetype)initWithVector:(ObjectVector<ObjectType>*)src;

@property(readonly) NSInteger length;
@property(readonly) NSArray<ObjectType>* array;
-(instancetype)initWithArray:(NSArray<ObjectType>*)array;

-(id)objectAtIndexedSubscript:(NSInteger)index;
-(ObjectType)get:(NSInteger)index;
-(void)updateArray:(NSArray<ObjectType>*)array;

@end
NS_ASSUME_NONNULL_END
