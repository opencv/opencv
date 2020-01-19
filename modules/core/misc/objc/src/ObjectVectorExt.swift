//
//  DoubleVectorExt.swift
//  InteropTest
//
//  Created by Giles Payne on 2020/01/04.
//  Copyright © 2020 Xtravision. All rights reserved.
//

import Foundation

extension ObjectVector : Sequence {
    public typealias Iterator = ObjectVectorIterator
    @objc public func makeIterator() -> ObjectVectorIterator {
        return ObjectVectorIterator(self as! ObjectVector<AnyObject>)
    }
}

@objc public class ObjectVectorIterator: NSObject, IteratorProtocol {
    public typealias Element = AnyObject
    let objectVector: ObjectVector<AnyObject>
    var pos = 0

    init(_ objectVector: ObjectVector<AnyObject>) {
        self.objectVector = objectVector
    }
    
    public func next() -> AnyObject? {
        guard pos >= 0 && pos < objectVector.length
            else { return nil }

        pos += 1
        return objectVector.get(pos - 1)
    }
}
