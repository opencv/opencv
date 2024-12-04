//
//  CvTypeExt.swift
//
//  Created by Giles Payne on 2020/01/19.
//

import Foundation

//NOTE: Type constants and related functions are ported from modules/core/include/opencv2/core/hal/interface.h

public extension CvType {
    static let CV_8U: Int32 = 0
    static let CV_8S: Int32 = 1
    static let CV_16U: Int32 = 2
    static let CV_16S: Int32 = 3
    static let CV_32S: Int32 = 4
    static let CV_32U: Int32 = 12
    static let CV_64U: Int32 = 10
    static let CV_64S: Int32 = 11
    static let CV_32F: Int32 = 5
    static let CV_64F: Int32 = 6
    static let CV_16F: Int32 = 7
    static let CV_16BF: Int32 = 8
    static let CV_Bool: Int32 = 9

    static let CV_8UC1: Int32 = CV_8UC(1)
    static let CV_8UC2: Int32 = CV_8UC(2)
    static let CV_8UC3: Int32 = CV_8UC(3)
    static let CV_8UC4: Int32 = CV_8UC(4)
    static let CV_8SC1: Int32 = CV_8SC(1)
    static let CV_8SC2: Int32 = CV_8SC(2)
    static let CV_8SC3: Int32 = CV_8SC(3)
    static let CV_8SC4: Int32 = CV_8SC(4)

    static let CV_16UC1: Int32 = CV_16UC(1)
    static let CV_16UC2: Int32 = CV_16UC(2)
    static let CV_16UC3: Int32 = CV_16UC(3)
    static let CV_16UC4: Int32 = CV_16UC(4)
    static let CV_16SC1: Int32 = CV_16SC(1)
    static let CV_16SC2: Int32 = CV_16SC(2)
    static let CV_16SC3: Int32 = CV_16SC(3)
    static let CV_16SC4: Int32 = CV_16SC(4)

    static let CV_32SC1: Int32 = CV_32SC(1)
    static let CV_32SC2: Int32 = CV_32SC(2)
    static let CV_32SC3: Int32 = CV_32SC(3)
    static let CV_32SC4: Int32 = CV_32SC(4)
    static let CV_32UC1: Int32 = CV_32UC(1)
    static let CV_32UC2: Int32 = CV_32UC(2)
    static let CV_32UC3: Int32 = CV_32UC(3)
    static let CV_32UC4: Int32 = CV_32UC(4)

    static let CV_64SC1: Int32 = CV_64SC(1)
    static let CV_64SC2: Int32 = CV_64SC(2)
    static let CV_64SC3: Int32 = CV_64SC(3)
    static let CV_64SC4: Int32 = CV_64SC(4)
    static let CV_64UC1: Int32 = CV_64UC(1)
    static let CV_64UC2: Int32 = CV_64UC(2)
    static let CV_64UC3: Int32 = CV_64UC(3)
    static let CV_64UC4: Int32 = CV_64UC(4)

    static let CV_32FC1: Int32 = CV_32FC(1)
    static let CV_32FC2: Int32 = CV_32FC(2)
    static let CV_32FC3: Int32 = CV_32FC(3)
    static let CV_32FC4: Int32 = CV_32FC(4)

    static let CV_64FC1: Int32 = CV_64FC(1)
    static let CV_64FC2: Int32 = CV_64FC(2)
    static let CV_64FC3: Int32 = CV_64FC(3)
    static let CV_64FC4: Int32 = CV_64FC(4)

    static let CV_16FC1: Int32 = CV_16FC(1)
    static let CV_16FC2: Int32 = CV_16FC(2)
    static let CV_16FC3: Int32 = CV_16FC(3)
    static let CV_16FC4: Int32 = CV_16FC(4)

    static let CV_16BFC1: Int32 = CV_16BFC(1)
    static let CV_16BFC2: Int32 = CV_16BFC(2)
    static let CV_16BFC3: Int32 = CV_16BFC(3)
    static let CV_16BFC4: Int32 = CV_16BFC(4)

    static let CV_BoolC1: Int32 = CV_BoolC(1)
    static let CV_BoolC2: Int32 = CV_BoolC(2)
    static let CV_BoolC3: Int32 = CV_BoolC(3)
    static let CV_BoolC4: Int32 = CV_BoolC(4)

    static let CV_CN_MAX = 128
    static let CV_CN_SHIFT = 5
    static let CV_DEPTH_MAX = 1 << CV_CN_SHIFT

    static func CV_8UC(_ channels:Int32) -> Int32 {
        return make(CV_8U, channels: channels)
    }

    static func CV_8SC(_ channels:Int32) -> Int32 {
        return make(CV_8S, channels: channels)
    }

    static func CV_16UC(_ channels:Int32) -> Int32 {
        return make(CV_16U, channels: channels)
    }

    static func CV_16SC(_ channels:Int32) -> Int32 {
        return make(CV_16S, channels: channels)
    }

    static func CV_32SC(_ channels:Int32) -> Int32 {
        return make(CV_32S, channels: channels)
    }

    static func CV_32UC(_ channels:Int32) -> Int32 {
        return make(CV_32U, channels: channels)
    }

    static func CV_64SC(_ channels:Int32) -> Int32 {
        return make(CV_64S, channels: channels)
    }

    static func CV_64UC(_ channels:Int32) -> Int32 {
        return make(CV_64U, channels: channels)
    }

    static func CV_32FC(_ channels:Int32) -> Int32 {
        return make(CV_32F, channels: channels)
    }

    static func CV_64FC(_ channels:Int32) -> Int32 {
        return make(CV_64F, channels: channels)
    }

    static func CV_16FC(_ channels:Int32) -> Int32 {
        return make(CV_16F, channels: channels)
    }

    static func CV_16BFC(_ channels:Int32) -> Int32 {
        return make(CV_16BF, channels: channels)
    }

    static func CV_BoolC(_ channels:Int32) -> Int32 {
        return make(CV_Bool, channels: channels)
    }
}
