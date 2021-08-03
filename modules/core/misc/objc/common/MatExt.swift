//
//  MatExt.swift
//
//  Created by Giles Payne on 2020/01/19.
//

import Foundation

let OpenCVErrorDomain = "OpenCVErrorDomain"

enum OpenCVError : Int {
  case IncompatibleDataType = 10001
  case IncompatibleBufferSize
}

func throwIncompatibleDataType(typeName: String) throws {
    throw NSError(
        domain: OpenCVErrorDomain,
        code: OpenCVError.IncompatibleDataType.rawValue,
        userInfo: [
            NSLocalizedDescriptionKey: "Incompatible Mat type \(typeName)"
        ]
    )
}

func throwIncompatibleBufferSize(count: Int, channels: Int32) throws {
    throw NSError(
        domain: OpenCVErrorDomain,
        code: OpenCVError.IncompatibleBufferSize.rawValue,
        userInfo: [
            NSLocalizedDescriptionKey: "Provided data element number \(count) should be multiple of the Mat channels count \(channels)"
        ]
    )
}

public typealias T2<T> = (T, T)
public typealias T3<T> = (T, T, T)
public typealias T4<T> = (T, T, T, T)

public extension Mat {

    convenience init(rows:Int32, cols:Int32, type:Int32, data:[Int8]) {
        let dataObject = data.withUnsafeBufferPointer { Data(buffer: $0) }
        self.init(rows: rows, cols: cols, type: type, data: dataObject)
    }

    convenience init(rows:Int32, cols:Int32, type:Int32, data:[Int8], step:Int) {
        let dataObject = data.withUnsafeBufferPointer { Data(buffer: $0) }
        self.init(rows: rows, cols: cols, type: type, data: dataObject, step:step)
    }

    @discardableResult func get(indices:[Int32], data:inout [Int8]) throws -> Int32 {
        let channels = CvType.channels(Int32(type()))
        if Int32(data.count) % channels != 0 {
            try throwIncompatibleBufferSize(count: data.count, channels: channels)
        } else if depth() != CvType.CV_8U && depth() != CvType.CV_8S {
            try throwIncompatibleDataType(typeName: CvType.type(toString: type()))
        }
        let count = Int32(data.count)
        return data.withUnsafeMutableBufferPointer { body in
            return __get(indices as [NSNumber], count: count, byteBuffer: body.baseAddress!)
        }
    }

    @discardableResult func get(indices:[Int32], data:inout [UInt8]) throws -> Int32 {
        let channels = CvType.channels(Int32(type()))
        if Int32(data.count) % channels != 0 {
            try throwIncompatibleBufferSize(count: data.count, channels: channels)
        } else if depth() != CvType.CV_8U {
            try throwIncompatibleDataType(typeName: CvType.type(toString: type()))
        }
        let count = Int32(data.count)
        return data.withUnsafeMutableBufferPointer { body in
            body.withMemoryRebound(to: Int8.self) { reboundBody in
                return __get(indices as [NSNumber], count: count, byteBuffer: reboundBody.baseAddress!)
            }
        }
    }

    @discardableResult func get(indices:[Int32], data:inout [Double]) throws -> Int32 {
        let channels = CvType.channels(Int32(type()))
        if Int32(data.count) % channels != 0 {
            try throwIncompatibleBufferSize(count: data.count, channels: channels)
        } else if depth() != CvType.CV_64F {
            try throwIncompatibleDataType(typeName: CvType.type(toString: type()))
        }
        let count = Int32(data.count)
        return data.withUnsafeMutableBufferPointer { body in
            return __get(indices as [NSNumber], count: count, doubleBuffer: body.baseAddress!)
        }
    }

    @discardableResult func get(indices:[Int32], data:inout [Float]) throws -> Int32 {
        let channels = CvType.channels(Int32(type()))
        if Int32(data.count) % channels != 0 {
            try throwIncompatibleBufferSize(count: data.count, channels: channels)
        } else if depth() != CvType.CV_32F {
            try throwIncompatibleDataType(typeName: CvType.type(toString: type()))
        }
        let count = Int32(data.count)
        return data.withUnsafeMutableBufferPointer { body in
            return __get(indices as [NSNumber], count: count, floatBuffer: body.baseAddress!)
        }
    }

    @discardableResult func get(indices:[Int32], data:inout [Int32]) throws -> Int32 {
        let channels = CvType.channels(Int32(type()))
        if Int32(data.count) % channels != 0 {
            try throwIncompatibleBufferSize(count: data.count, channels: channels)
        } else if depth() != CvType.CV_32S {
            try throwIncompatibleDataType(typeName: CvType.type(toString: type()))
        }
        let count = Int32(data.count)
        return data.withUnsafeMutableBufferPointer { body in
            return __get(indices as [NSNumber], count: count, intBuffer: body.baseAddress!)
        }
    }

    @discardableResult func get(indices:[Int32], data:inout [Int16]) throws -> Int32 {
        let channels = CvType.channels(Int32(type()))
        if Int32(data.count) % channels != 0 {
            try throwIncompatibleBufferSize(count: data.count, channels: channels)
        } else if depth() != CvType.CV_16U && depth() != CvType.CV_16S {
            try throwIncompatibleDataType(typeName: CvType.type(toString: type()))
        }
        let count = Int32(data.count)
        return data.withUnsafeMutableBufferPointer { body in
            return __get(indices as [NSNumber], count: count, shortBuffer: body.baseAddress!)
        }
    }

    @discardableResult func get(indices:[Int32], data:inout [UInt16]) throws -> Int32 {
        let channels = CvType.channels(Int32(type()))
        if Int32(data.count) % channels != 0 {
            try throwIncompatibleBufferSize(count: data.count, channels: channels)
        } else if depth() != CvType.CV_16U {
            try throwIncompatibleDataType(typeName: CvType.type(toString: type()))
        }
        let count = Int32(data.count)
        return data.withUnsafeMutableBufferPointer { body in
            body.withMemoryRebound(to: Int16.self) { reboundBody in
                return __get(indices as [NSNumber], count: count, shortBuffer: reboundBody.baseAddress!)
            }
        }
    }

    @discardableResult func get(row: Int32, col: Int32, data:inout [Int8]) throws -> Int32 {
        return try get(indices: [row, col], data: &data)
    }

    @discardableResult func get(row: Int32, col: Int32, data:inout [UInt8]) throws -> Int32 {
        return try get(indices: [row, col], data: &data)
    }

    @discardableResult func get(row: Int32, col: Int32, data:inout [Double]) throws -> Int32 {
        return try get(indices: [row, col], data: &data)
    }

    @discardableResult func get(row: Int32, col: Int32, data:inout [Float]) throws -> Int32 {
        return try get(indices: [row, col], data: &data)
    }

    @discardableResult func get(row: Int32, col: Int32, data:inout [Int32]) throws -> Int32 {
        return try get(indices: [row, col], data: &data)
    }

    @discardableResult func get(row: Int32, col: Int32, data:inout [Int16]) throws -> Int32 {
        return try get(indices: [row, col], data: &data)
    }

    @discardableResult func get(row: Int32, col: Int32, data:inout [UInt16]) throws -> Int32 {
        return try get(indices: [row, col], data: &data)
    }

    @discardableResult func put(indices:[Int32], data:[Int8]) throws -> Int32 {
        let channels = CvType.channels(Int32(type()))
        if Int32(data.count) % channels != 0 {
            try throwIncompatibleBufferSize(count: data.count, channels: channels)
        } else if depth() != CvType.CV_8U && depth() != CvType.CV_8S {
            try throwIncompatibleDataType(typeName: CvType.type(toString: type()))
        }
        let count = Int32(data.count)
        return data.withUnsafeBufferPointer { body in
            return __put(indices as [NSNumber], count: count, byteBuffer: body.baseAddress!)
        }
    }

    @discardableResult func put(indices:[Int32], data:[UInt8]) throws -> Int32 {
        let channels = CvType.channels(Int32(type()))
        if Int32(data.count) % channels != 0 {
            try throwIncompatibleBufferSize(count: data.count, channels: channels)
        } else if depth() != CvType.CV_8U {
            try throwIncompatibleDataType(typeName: CvType.type(toString: type()))
        }
        let count = Int32(data.count)
        return data.withUnsafeBufferPointer { body in
            body.withMemoryRebound(to: Int8.self) { reboundBody in
                return __put(indices as [NSNumber], count: count, byteBuffer: reboundBody.baseAddress!)
            }
        }
    }

    @discardableResult func put(indices:[Int32], data:[Int8], offset: Int, length: Int32) throws -> Int32 {
        let channels = CvType.channels(Int32(type()))
        if Int32(data.count) % channels != 0 {
            try throwIncompatibleBufferSize(count: data.count, channels: channels)
        } else if depth() != CvType.CV_8U && depth() != CvType.CV_8S {
            try throwIncompatibleDataType(typeName: CvType.type(toString: type()))
        }
        return data.withUnsafeBufferPointer { body in
            return __put(indices as [NSNumber], count: length, byteBuffer: body.baseAddress! + offset)
        }
    }

    // unlike other put:indices:data functions this one (with [Double]) should convert input values to correct type
    @discardableResult func put(indices:[Int32], data:[Double]) throws -> Int32 {
        let channels = CvType.channels(Int32(type()))
        if Int32(data.count) % channels != 0 {
            try throwIncompatibleBufferSize(count: data.count, channels: channels)
        }
        if depth() == CvType.CV_64F {
            let count = Int32(data.count)
            return data.withUnsafeBufferPointer { body in
                return __put(indices as [NSNumber], count: count, doubleBuffer: body.baseAddress!)
            }
        } else {
            return __put(indices as [NSNumber], data: data as [NSNumber])
        }
    }

    @discardableResult func put(indices:[Int32], data:[Float]) throws -> Int32 {
        let channels = CvType.channels(Int32(type()))
        if Int32(data.count) % channels != 0 {
            try throwIncompatibleBufferSize(count: data.count, channels: channels)
        } else if depth() != CvType.CV_32F {
            try throwIncompatibleDataType(typeName: CvType.type(toString: type()))
        }
        let count = Int32(data.count)
        return data.withUnsafeBufferPointer { body in
            return __put(indices as [NSNumber], count: count, floatBuffer: body.baseAddress!)
        }
    }

    @discardableResult func put(indices:[Int32], data:[Int32]) throws -> Int32 {
        let channels = CvType.channels(Int32(type()))
        if Int32(data.count) % channels != 0 {
            try throwIncompatibleBufferSize(count: data.count, channels: channels)
        } else if depth() != CvType.CV_32S {
            try throwIncompatibleDataType(typeName: CvType.type(toString: type()))
        }
        let count = Int32(data.count)
        return data.withUnsafeBufferPointer { body in
            return __put(indices as [NSNumber], count: count, intBuffer: body.baseAddress!)
        }
    }

    @discardableResult func put(indices:[Int32], data:[Int16]) throws -> Int32 {
        let channels = CvType.channels(Int32(type()))
        if Int32(data.count) % channels != 0 {
            try throwIncompatibleBufferSize(count: data.count, channels: channels)
        } else if depth() != CvType.CV_16U && depth() != CvType.CV_16S {
            try throwIncompatibleDataType(typeName: CvType.type(toString: type()))
        }
        let count = Int32(data.count)
        return data.withUnsafeBufferPointer { body in
            return __put(indices as [NSNumber], count: count, shortBuffer: body.baseAddress!)
        }
    }

    @discardableResult func put(indices:[Int32], data:[UInt16]) throws -> Int32 {
        let channels = CvType.channels(Int32(type()))
        if Int32(data.count) % channels != 0 {
            try throwIncompatibleBufferSize(count: data.count, channels: channels)
        } else if depth() != CvType.CV_16U {
            try throwIncompatibleDataType(typeName: CvType.type(toString: type()))
        }
        let count = Int32(data.count)
        return data.withUnsafeBufferPointer { body in
            body.withMemoryRebound(to: Int16.self) { reboundBody in
                return __put(indices as [NSNumber], count: count, shortBuffer: reboundBody.baseAddress!)
            }
        }
    }

    @discardableResult func put(row: Int32, col: Int32, data:[Int8]) throws -> Int32 {
        return try put(indices: [row, col], data: data)
    }

    @discardableResult func put(row: Int32, col: Int32, data:[UInt8]) throws -> Int32 {
        return try put(indices: [row, col], data: data)
    }

    @discardableResult func put(row: Int32, col: Int32, data: [Int8], offset: Int, length: Int32) throws -> Int32 {
        return try put(indices: [row, col], data: data, offset: offset, length: length)
    }

    @discardableResult func put(row: Int32, col: Int32, data: [Double]) throws -> Int32 {
        return try put(indices: [row, col], data: data)
    }

    @discardableResult func put(row: Int32, col: Int32, data: [Float]) throws -> Int32 {
        return try put(indices: [row, col], data: data)
    }

    @discardableResult func put(row: Int32, col: Int32, data: [Int32]) throws -> Int32 {
        return try put(indices: [row, col], data: data)
    }

    @discardableResult func put(row: Int32, col: Int32, data: [Int16]) throws -> Int32 {
        return try put(indices: [row, col], data: data)
    }

    @discardableResult func put(row: Int32, col: Int32, data: [UInt16]) throws -> Int32 {
        return try put(indices: [row, col], data: data)
    }

    @discardableResult func get(row: Int32, col: Int32) -> [Double] {
        return get(indices: [row, col])
    }

    @discardableResult func get(indices: [Int32]) -> [Double] {
        return __get(indices as [NSNumber]) as! [Double]
    }
}

public protocol Atable {
    static func getAt(m: Mat, indices:[Int32]) -> Self
    static func putAt(m: Mat, indices:[Int32], v: Self)
    static func getAt2c(m: Mat, indices:[Int32]) -> (Self, Self)
    static func putAt2c(m: Mat, indices:[Int32], v: (Self, Self))
    static func getAt3c(m: Mat, indices:[Int32]) -> (Self, Self, Self)
    static func putAt3c(m: Mat, indices:[Int32], v: (Self, Self, Self))
    static func getAt4c(m: Mat, indices:[Int32]) -> (Self, Self, Self, Self)
    static func putAt4c(m: Mat, indices:[Int32], v: (Self, Self, Self, Self))
}

public class MatAt<N: Atable> {

    init(mat: Mat, indices: [Int32]) {
        self.mat = mat
        self.indices = indices
    }

    private let mat: Mat
    private let indices: [Int32]
    public var v: N {
        get {
            return N.getAt(m: mat, indices: indices)
        }
        set(value) {
            N.putAt(m: mat, indices: indices, v: value)
        }
    }
    public var v2c: (N, N) {
        get {
            return N.getAt2c(m: mat, indices: indices)
        }
        set(value) {
            N.putAt2c(m: mat, indices: indices, v: value)
        }
    }
    public var v3c: (N, N, N) {
        get {
            return N.getAt3c(m: mat, indices: indices)
        }
        set(value) {
            N.putAt3c(m: mat, indices: indices, v: value)
        }
    }
    public var v4c: (N, N, N, N) {
        get {
            return N.getAt4c(m: mat, indices: indices)
        }
        set(value) {
            N.putAt4c(m: mat, indices: indices, v: value)
        }
    }
}

extension UInt8: Atable {
    public static func getAt(m: Mat, indices:[Int32]) -> UInt8 {
        var tmp = [UInt8](repeating: 0, count: 1)
        try! m.get(indices: indices, data: &tmp)
        return tmp[0]
    }

    public static func putAt(m: Mat, indices: [Int32], v: UInt8) {
        let tmp = [v]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt2c(m: Mat, indices:[Int32]) -> (UInt8, UInt8) {
        var tmp = [UInt8](repeating: 0, count: 2)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1])
    }

    public static func putAt2c(m: Mat, indices: [Int32], v: (UInt8, UInt8)) {
        let tmp = [v.0, v.1]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt3c(m: Mat, indices:[Int32]) -> (UInt8, UInt8, UInt8) {
        var tmp = [UInt8](repeating: 0, count: 3)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1], tmp[2])
    }

    public static func putAt3c(m: Mat, indices: [Int32], v: (UInt8, UInt8, UInt8)) {
        let tmp = [v.0, v.1, v.2]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt4c(m: Mat, indices:[Int32]) -> (UInt8, UInt8, UInt8, UInt8) {
        var tmp = [UInt8](repeating: 0, count: 4)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1], tmp[2], tmp[3])
    }

    public static func putAt4c(m: Mat, indices: [Int32], v: (UInt8, UInt8, UInt8, UInt8)) {
        let tmp = [v.0, v.1, v.2, v.3]
        try! m.put(indices: indices, data: tmp)
    }
}

extension Int8: Atable {
    public static func getAt(m: Mat, indices:[Int32]) -> Int8 {
        var tmp = [Int8](repeating: 0, count: 1)
        try! m.get(indices: indices, data: &tmp)
        return tmp[0]
    }

    public static func putAt(m: Mat, indices: [Int32], v: Int8) {
        let tmp = [v]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt2c(m: Mat, indices:[Int32]) -> (Int8, Int8) {
        var tmp = [Int8](repeating: 0, count: 2)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1])
    }

    public static func putAt2c(m: Mat, indices: [Int32], v: (Int8, Int8)) {
        let tmp = [v.0, v.1]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt3c(m: Mat, indices:[Int32]) -> (Int8, Int8, Int8) {
        var tmp = [Int8](repeating: 0, count: 3)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1], tmp[2])
    }

    public static func putAt3c(m: Mat, indices: [Int32], v: (Int8, Int8, Int8)) {
        let tmp = [v.0, v.1, v.2]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt4c(m: Mat, indices:[Int32]) -> (Int8, Int8, Int8, Int8) {
        var tmp = [Int8](repeating: 0, count: 4)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1], tmp[2], tmp[3])
    }

    public static func putAt4c(m: Mat, indices: [Int32], v: (Int8, Int8, Int8, Int8)) {
        let tmp = [v.0, v.1, v.2, v.3]
        try! m.put(indices: indices, data: tmp)
    }
}

extension Double: Atable {
    public static func getAt(m: Mat, indices:[Int32]) -> Double {
        var tmp = [Double](repeating: 0, count: 1)
        try! m.get(indices: indices, data: &tmp)
        return tmp[0]
    }

    public static func putAt(m: Mat, indices: [Int32], v: Double) {
        let tmp = [v]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt2c(m: Mat, indices:[Int32]) -> (Double, Double) {
        var tmp = [Double](repeating: 0, count: 2)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1])
    }

    public static func putAt2c(m: Mat, indices: [Int32], v: (Double, Double)) {
        let tmp = [v.0, v.1]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt3c(m: Mat, indices:[Int32]) -> (Double, Double, Double) {
        var tmp = [Double](repeating: 0, count: 3)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1], tmp[2])
    }

    public static func putAt3c(m: Mat, indices: [Int32], v: (Double, Double, Double)) {
        let tmp = [v.0, v.1, v.2]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt4c(m: Mat, indices:[Int32]) -> (Double, Double, Double, Double) {
        var tmp = [Double](repeating: 0, count: 4)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1], tmp[2], tmp[3])
    }

    public static func putAt4c(m: Mat, indices: [Int32], v: (Double, Double, Double, Double)) {
        let tmp = [v.0, v.1, v.2, v.3]
        try! m.put(indices: indices, data: tmp)
    }
}

extension Float: Atable {
    public static func getAt(m: Mat, indices:[Int32]) -> Float {
        var tmp = [Float](repeating: 0, count: 1)
        try! m.get(indices: indices, data: &tmp)
        return tmp[0]
    }

    public static func putAt(m: Mat, indices: [Int32], v: Float) {
        let tmp = [v]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt2c(m: Mat, indices:[Int32]) -> (Float, Float) {
        var tmp = [Float](repeating: 0, count: 2)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1])
    }

    public static func putAt2c(m: Mat, indices: [Int32], v: (Float, Float)) {
        let tmp = [v.0, v.1]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt3c(m: Mat, indices:[Int32]) -> (Float, Float, Float) {
        var tmp = [Float](repeating: 0, count: 3)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1], tmp[2])
    }

    public static func putAt3c(m: Mat, indices: [Int32], v: (Float, Float, Float)) {
        let tmp = [v.0, v.1, v.2]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt4c(m: Mat, indices:[Int32]) -> (Float, Float, Float, Float) {
        var tmp = [Float](repeating: 0, count: 4)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1], tmp[2], tmp[3])
    }

    public static func putAt4c(m: Mat, indices: [Int32], v: (Float, Float, Float, Float)) {
        let tmp = [v.0, v.1, v.2, v.3]
        try! m.put(indices: indices, data: tmp)
    }
}

extension Int32: Atable {
    public static func getAt(m: Mat, indices:[Int32]) -> Int32 {
        var tmp = [Int32](repeating: 0, count: 1)
        try! m.get(indices: indices, data: &tmp)
        return tmp[0]
    }

    public static func putAt(m: Mat, indices: [Int32], v: Int32) {
        let tmp = [v]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt2c(m: Mat, indices:[Int32]) -> (Int32, Int32) {
        var tmp = [Int32](repeating: 0, count: 2)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1])
    }

    public static func putAt2c(m: Mat, indices: [Int32], v: (Int32, Int32)) {
        let tmp = [v.0, v.1]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt3c(m: Mat, indices:[Int32]) -> (Int32, Int32, Int32) {
        var tmp = [Int32](repeating: 0, count: 3)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1], tmp[2])
    }

    public static func putAt3c(m: Mat, indices: [Int32], v: (Int32, Int32, Int32)) {
        let tmp = [v.0, v.1, v.2]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt4c(m: Mat, indices:[Int32]) -> (Int32, Int32, Int32, Int32) {
        var tmp = [Int32](repeating: 0, count: 4)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1], tmp[2], tmp[3])
    }

    public static func putAt4c(m: Mat, indices: [Int32], v: (Int32, Int32, Int32, Int32)) {
        let tmp = [v.0, v.1, v.2, v.3]
        try! m.put(indices: indices, data: tmp)
    }
}

extension UInt16: Atable {
    public static func getAt(m: Mat, indices:[Int32]) -> UInt16 {
        var tmp = [UInt16](repeating: 0, count: 1)
        try! m.get(indices: indices, data: &tmp)
        return tmp[0]
    }

    public static func putAt(m: Mat, indices: [Int32], v: UInt16) {
        let tmp = [v]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt2c(m: Mat, indices:[Int32]) -> (UInt16, UInt16) {
        var tmp = [UInt16](repeating: 0, count: 2)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1])
    }

    public static func putAt2c(m: Mat, indices: [Int32], v: (UInt16, UInt16)) {
        let tmp = [v.0, v.1]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt3c(m: Mat, indices:[Int32]) -> (UInt16, UInt16, UInt16) {
        var tmp = [UInt16](repeating: 0, count: 3)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1], tmp[2])
    }

    public static func putAt3c(m: Mat, indices: [Int32], v: (UInt16, UInt16, UInt16)) {
        let tmp = [v.0, v.1, v.2]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt4c(m: Mat, indices:[Int32]) -> (UInt16, UInt16, UInt16, UInt16) {
        var tmp = [UInt16](repeating: 0, count: 4)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1], tmp[2], tmp[3])
    }

    public static func putAt4c(m: Mat, indices: [Int32], v: (UInt16, UInt16, UInt16, UInt16)) {
        let tmp = [v.0, v.1, v.2, v.3]
        try! m.put(indices: indices, data: tmp)
    }
}

extension Int16: Atable {
    public static func getAt(m: Mat, indices:[Int32]) -> Int16 {
        var tmp = [Int16](repeating: 0, count: 1)
        try! m.get(indices: indices, data: &tmp)
        return tmp[0]
    }

    public static func putAt(m: Mat, indices: [Int32], v: Int16) {
        let tmp = [v]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt2c(m: Mat, indices:[Int32]) -> (Int16, Int16) {
        var tmp = [Int16](repeating: 0, count: 2)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1])
    }

    public static func putAt2c(m: Mat, indices: [Int32], v: (Int16, Int16)) {
        let tmp = [v.0, v.1]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt3c(m: Mat, indices:[Int32]) -> (Int16, Int16, Int16) {
        var tmp = [Int16](repeating: 0, count: 3)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1], tmp[2])
    }

    public static func putAt3c(m: Mat, indices: [Int32], v: (Int16, Int16, Int16)) {
        let tmp = [v.0, v.1, v.2]
        try! m.put(indices: indices, data: tmp)
    }

    public static func getAt4c(m: Mat, indices:[Int32]) -> (Int16, Int16, Int16, Int16) {
        var tmp = [Int16](repeating: 0, count: 4)
        try! m.get(indices: indices, data: &tmp)
        return (tmp[0], tmp[1], tmp[2], tmp[3])
    }

    public static func putAt4c(m: Mat, indices: [Int32], v: (Int16, Int16, Int16, Int16)) {
        let tmp = [v.0, v.1, v.2, v.3]
        try! m.put(indices: indices, data: tmp)
    }
}

/***
 *  Example use:
 *
 *  let elemantVal: UInt8 = mat.at(row: 50, col: 50).v
 *  mat.at(row: 50, col: 50).v = 245
 *
 */
public extension Mat {
    func at<N: Atable>(row: Int32, col: Int32) -> MatAt<N> {
        return MatAt(mat: self, indices: [row, col])
    }

    func at<N: Atable>(indices:[Int32]) -> MatAt<N> {
        return MatAt(mat: self, indices: indices)
    }
}
