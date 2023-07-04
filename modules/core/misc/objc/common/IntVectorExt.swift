//
//  IntVectorExt.swift
//
//  Created by Giles Payne on 2020/01/04.
//

import Foundation

public extension IntVector {
    convenience init(_ array:[Int32]) {
        let data = array.withUnsafeBufferPointer { Data(buffer: $0) }
        self.init(data:data);
    }

    subscript(index: Int) -> Int32 {
        get {
            return self.get(index)
        }
    }

    var array: [Int32] {
        get {
            var ret = Array<Int32>(repeating: 0, count: data.count/MemoryLayout<Int32>.stride)
            _ = ret.withUnsafeMutableBytes { data.copyBytes(to: $0) }
            return ret
        }
    }
}

extension IntVector : Sequence {
    public typealias Iterator = IntVectorIterator
    public func makeIterator() -> IntVectorIterator {
        return IntVectorIterator(self)
    }
}

public struct IntVectorIterator: IteratorProtocol {
    public typealias Element = Int32
    let intVector: IntVector
    var pos = 0

    init(_ intVector: IntVector) {
        self.intVector = intVector
    }

    mutating public func next() -> Int32? {
        guard pos >= 0 && pos < intVector.length
            else { return nil }

        pos += 1
        return intVector.get(pos - 1)
    }
}
