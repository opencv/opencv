//
//  FloatVectorExt.swift
//
//  Created by Giles Payne on 2020/01/04.
//

import Foundation

public extension FloatVector {
    convenience init(_ array:[Float]) {
        let data = array.withUnsafeBufferPointer { Data(buffer: $0) }
        self.init(data:data);
    }

    subscript(index: Int) -> Float {
        get {
            return self.get(index)
        }
    }

    var array: [Float] {
        get {
            var ret = Array<Float>(repeating: 0, count: data.count/MemoryLayout<Float>.stride)
            _ = ret.withUnsafeMutableBytes { data.copyBytes(to: $0) }
            return ret
        }
    }
}

extension FloatVector : Sequence {
    public typealias Iterator = FloatVectorIterator
    public func makeIterator() -> FloatVectorIterator {
        return FloatVectorIterator(self)
    }
}

public struct FloatVectorIterator: IteratorProtocol {
    public typealias Element = Float
    let floatVector: FloatVector
    var pos = 0

    init(_ floatVector: FloatVector) {
        self.floatVector = floatVector
    }

    mutating public func next() -> Float? {
        guard pos >= 0 && pos < floatVector.length
            else { return nil }

        pos += 1
        return floatVector.get(pos - 1)
    }
}
