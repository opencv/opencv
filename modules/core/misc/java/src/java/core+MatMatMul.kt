package org.opencv.core

operator fun Mat.times(other: Mat): Mat = this.matMul(other)
