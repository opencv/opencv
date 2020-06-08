//
//  ByteVectorExt.swift
//
//  Created by Giles Payne on 2020/01/04.
//

import Foundation

public extension ByteVector {
    convenience init(_ array:[Int8]) {
        let data = array.withUnsafeBufferPointer { Data(buffer: $0) }
        self.init(data:data);
    }

    subscript(index: Int) -> Int8 {
        get {
            return self.get(index)
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
    let byteVector: ByteVector
    var pos = 0

    init(_ byteVector: ByteVector) {
        self.byteVector = byteVector
    }

    mutating public func next() -> Int8? {
        guard pos >= 0 && pos < byteVector.length
            else { return nil }

        pos += 1
        return byteVector.get(pos - 1)
    }
}
