/*
 * TestBar.h
 *
 *  Created on: Jul 17, 2010
 *      Author: ethan
 */

#ifndef TESTBAR_H_
#define TESTBAR_H_

#include "image_pool.h"

struct FooBarStruct {

	int pool_image_count(image_pool* pool){
		return pool->getCount();
	}

};

class BarBar{
public:
	void crazy();
};

#endif /* TESTBAR_H_ */
