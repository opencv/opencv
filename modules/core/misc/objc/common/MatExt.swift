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

    @discardableResult func get(row: Int32, col: Int32, data:inout [Int8]) throws -> Int32 {
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

    @discardableResult func put(row: Int32, col: Int32, data:[Int8]) throws -> Int32 {
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

    @discardableResult func get(row: Int32, col: Int32) -> [Double] {
        return get(indices: [row, col])
    }

    @discardableResult func get(indices: [Int32]) -> [Double] {
        return __get(indices as [NSNumber]) as! [Double]
    }
}
