//
//  DoubleVectorExt.swift
//
//  Created by Giles Payne on 2020/01/04.
//

import Foundation

public extension DoubleVector {
    convenience init(_ array:[Double]) {
        let data = array.withUnsafeBufferPointer { Data(buffer: $0) }
        self.init(data:data);
    }

    subscript(index: Int) -> Double {
        get {
            return self.get(index)
        }
    }

    var array: [Double] {
        get {
            var ret = Array<Double>(repeating: 0, count: data.count/MemoryLayout<Double>.stride)
            _ = ret.withUnsafeMutableBytes { data.copyBytes(to: $0) }
            return ret
        }
    }
}

extension DoubleVector : Sequence {
    public typealias Iterator = DoubleVectorIterator
    public func makeIterator() -> DoubleVectorIterator {
        return DoubleVectorIterator(self)
    }
}

public struct DoubleVectorIterator: IteratorProtocol {
    public typealias Element = Double
    let doubleVector: DoubleVector
    var pos = 0

    init(_ doubleVector: DoubleVector) {
        self.doubleVector = doubleVector
    }

    mutating public func next() -> Double? {
        guard pos >= 0 && pos < doubleVector.length
            else { return nil }

        pos += 1
        return doubleVector.get(pos - 1)
    }
}
