//
//  ByteVectorExt.swift
//
//  Created by Giles Payne on 2020/01/04.
//

import Foundation

extension ByteVector {
    convenience init(_ array:[Int8]) {
        let data = Data(buffer: UnsafeBufferPointer(start: array, count: array.count))
        self.init(data:data);
    }

    subscript(index: Int) -> Int8 {
        get {
            self.get(index)
        }
    }
    
    var array: [Int8] {
        get {
            var ret = Array<Int8>(repeating: 0, count: data.count/MemoryLayout<Int8>.stride)
            _ = ret.withUnsafeMutableBytes { data.copyBytes(to: $0) }
            return ret
        }
    }
}

extension ByteVector : Sequence {
    public typealias Iterator = ByteVectorIterator
    public func makeIterator() -> ByteVectorIterator {
        return ByteVectorIterator(self)
    }
}

public struct ByteVectorIterator: IteratorProtocol {
    public typealias Element = Int8
    let floatVector: ByteVector
    var pos = 0

    init(_ floatVector: ByteVector) {
        self.floatVector = floatVector
    }
    
    mutating public func next() -> Int8? {
        guard pos >= 0 && pos < floatVector.length
            else { return nil }

        pos += 1
        return floatVector.get(pos - 1)
    }
}
