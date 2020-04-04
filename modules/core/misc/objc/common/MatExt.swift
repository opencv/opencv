//
//  MatExt.swift
//
//  Created by Giles Payne on 2020/01/19.
//

import Foundation

extension Mat {

    public convenience init(rows:Int32, cols:Int32, type:Int32, buffer:[Int8]) {
        let data = Data(buffer: UnsafeBufferPointer(start: buffer, count: buffer.count))
        self.init(rows: rows, cols: cols, type: type, data: data)
    }


    public convenience init(rows:Int32, cols:Int32, type:Int32, buffer:[Int8], step:Int) {
        let data = Data(buffer: UnsafeBufferPointer(start: buffer, count: buffer.count))
        self.init(rows: rows, cols: cols, type: type, data: data, step:step)
    }

}
