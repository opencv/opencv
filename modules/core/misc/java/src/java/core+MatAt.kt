package org.opencv.core

import org.opencv.core.Mat.*
import java.lang.RuntimeException

/***
 *  Example use:
 *
 *  val (b, g, r) = mat.at<UByte>(50, 50).v3c
 *  mat.at<UByte>(50, 50).val = T3(245u, 113u, 34u)
 *
 */
@Suppress("UNCHECKED_CAST")
inline fun <reified T> Mat.at(row: Int, col: Int) : Atable<T> =
    when (T::class) {
        Byte::class, Double::class, Float::class, Int::class, Short::class -> this.at(
            T::class.java,
            row,
            col
        )
        UByte::class -> AtableUByte(this, row, col) as Atable<T>
        else -> throw RuntimeException("Unsupported class type")
    }

@Suppress("UNCHECKED_CAST")
inline fun <reified T> Mat.at(idx: IntArray) : Atable<T> =
    when (T::class) {
        Byte::class, Double::class, Float::class, Int::class, Short::class -> this.at(
            T::class.java,
            idx
        )
        UByte::class -> AtableUByte(this, idx) as Atable<T>
        else -> throw RuntimeException("Unsupported class type")
    }

class AtableUByte(val mat: Mat, val indices: IntArray): Atable<UByte> {

    constructor(mat: Mat, row: Int, col: Int) : this(mat, intArrayOf(row, col))

    override fun getV(): UByte {
        val data = ByteArray(1)
        mat[indices, data]
        return data[0].toUByte()
    }

    override fun setV(v: UByte) {
        val data = byteArrayOf(v.toByte())
        mat.put(indices, data)
    }

    override fun getV2c(): Tuple2<UByte> {
        val data = ByteArray(2)
        mat[indices, data]
        return Tuple2(data[0].toUByte(), data[1].toUByte())
    }

    override fun setV2c(v: Tuple2<UByte>) {
        val data = byteArrayOf(v._0.toByte(), v._1.toByte())
        mat.put(indices, data)
    }

    override fun getV3c(): Tuple3<UByte> {
        val data = ByteArray(3)
        mat[indices, data]
        return Tuple3(data[0].toUByte(), data[1].toUByte(), data[2].toUByte())
    }

    override fun setV3c(v: Tuple3<UByte>) {
        val data = byteArrayOf(v._0.toByte(), v._1.toByte(), v._2.toByte())
        mat.put(indices, data)
    }

    override fun getV4c(): Tuple4<UByte> {
        val data = ByteArray(4)
        mat[indices, data]
        return Tuple4(data[0].toUByte(), data[1].toUByte(), data[2].toUByte(), data[3].toUByte())
    }

    override fun setV4c(v: Tuple4<UByte>) {
        val data = byteArrayOf(v._0.toByte(), v._1.toByte(), v._2.toByte(), v._3.toByte())
        mat.put(indices, data)
    }
}

operator fun <T> Tuple2<T>.component1(): T = this._0
operator fun <T> Tuple2<T>.component2(): T = this._1

operator fun <T> Tuple3<T>.component1(): T = this._0
operator fun <T> Tuple3<T>.component2(): T = this._1
operator fun <T> Tuple3<T>.component3(): T = this._2

operator fun <T> Tuple4<T>.component1(): T = this._0
operator fun <T> Tuple4<T>.component2(): T = this._1
operator fun <T> Tuple4<T>.component3(): T = this._2
operator fun <T> Tuple4<T>.component4(): T = this._3

fun <T> T2(_0: T, _1: T) : Tuple2<T> = Tuple2(_0, _1)
fun <T> T3(_0: T, _1: T, _2: T) : Tuple3<T> = Tuple3(_0, _1, _2)
fun <T> T4(_0: T, _1: T, _2: T, _3: T) : Tuple4<T> = Tuple4(_0, _1, _2, _3)
