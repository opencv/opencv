//
//  CVObjcUtil.h
//
//  Created by Giles Payne on 2020/01/02.
//

#pragma once

typedef union { double d; int64_t l; } V64;
typedef union { float f; int32_t i; } V32;

#define DOUBLE_TO_BITS(x)  ((V64){ .d = x }).l
#define FLOAT_TO_BITS(x)  ((V32){ .f = x }).i

#ifdef __cplusplus
#import <vector>

template <typename CV, typename OBJC> std::vector<CV> objc2cv(NSArray<OBJC*>* _Nonnull array, CV& (* _Nonnull converter)(OBJC* _Nonnull)) {
    std::vector<CV> ret;
    for (OBJC* obj in array) {
        ret.push_back(converter(obj));
    }
    return ret;
}

#define OBJC2CV(CV_CLASS, OBJC_CLASS, v, a) \
    std::vector<CV_CLASS> v = objc2cv<CV_CLASS, OBJC_CLASS>(a, [](OBJC_CLASS* objc) -> CV_CLASS& { return objc.nativeRef; })

template <typename CV, typename CV_RAW, typename OBJC> std::vector<CV> objc2cv_c(NSArray<OBJC*>* _Nonnull array, CV_RAW& (* _Nonnull converter)(OBJC* _Nonnull)) {
    std::vector<CV> ret;
    for (OBJC* obj in array) {
        CV tmp = CV(converter(obj));
        ret.push_back(tmp);
    }
    return ret;
}

#define OBJC2CV_C(CV_CLASS, CV_CLASS_RAW, OBJC_CLASS, v, a) \
    std::vector<CV_CLASS> v = objc2cv_c<CV_CLASS, CV_CLASS_RAW, OBJC_CLASS>(a, [](OBJC_CLASS* objc) -> CV_CLASS_RAW& { return objc.nativeRef; })

template <typename CV, typename OBJC> void cv2objc(std::vector<CV>& vector, NSMutableArray<OBJC*>* _Nonnull array, OBJC* _Nonnull (* _Nonnull converter)(CV&)) {
    [array removeAllObjects];
    for (size_t index = 0; index < vector.size(); index++) {
        [array addObject:converter(vector[index])];
    }
}

#define CV2OBJC(CV_CLASS, OBJC_CLASS, v, a) \
    cv2objc<CV_CLASS, OBJC_CLASS>(v, a, [](CV_CLASS& cv) -> OBJC_CLASS* { return [OBJC_CLASS fromNative:cv]; })

template <typename CV, typename OBJC> std::vector<std::vector<CV>> objc2cv2(NSArray<NSArray<OBJC*>*>* _Nonnull array, CV& (* _Nonnull converter)(OBJC* _Nonnull)) {
    std::vector<std::vector<CV>> ret;
    for (NSArray<OBJC*>* innerArray in array) {
        std::vector<CV> innerVector;
        for (OBJC* obj in innerArray) {
            innerVector.push_back(converter(obj));
        }
        ret.push_back(innerVector);
    }
    return ret;
}

#define OBJC2CV2(CV_CLASS, OBJC_CLASS, v, a) \
    std::vector<std::vector<CV_CLASS>> v = objc2cv2<CV_CLASS, OBJC_CLASS>(a, [](OBJC_CLASS* objc) -> CV_CLASS& { return objc.nativeRef; })

template <typename CV, typename CV_RAW, typename OBJC> std::vector<std::vector<CV>> objc2cv2_c(NSArray<NSArray<OBJC*>*>* _Nonnull array, CV_RAW& (* _Nonnull converter)(OBJC* _Nonnull)) {
    std::vector<std::vector<CV>> ret;
    for (NSArray<OBJC*>* innerArray in array) {
        std::vector<CV> innerVector;
        for (OBJC* obj in innerArray) {
            CV tmp = CV(converter(obj));
            innerVector.push_back(tmp);
        }
        ret.push_back(innerVector);
    }
    return ret;
}

#define OBJC2CV2_C(CV_CLASS, CV_CLASS_RAW, OBJC_CLASS, v, a) \
    std::vector<std::vector<CV_CLASS>> v = objc2cv2_c<CV_CLASS, CV_CLASS_RAW, OBJC_CLASS>(a, [](OBJC_CLASS* objc) -> CV_CLASS_RAW& { return objc.nativeRef; })

template <typename CV, typename OBJC> void cv2objc2(std::vector<std::vector<CV>>& vector, NSMutableArray<NSMutableArray<OBJC*>*>* _Nonnull array, OBJC* _Nonnull (* _Nonnull converter)(CV&)) {
    [array removeAllObjects];
    for (size_t index = 0; index < vector.size(); index++) {
        std::vector<CV>& innerVector = vector[index];
        NSMutableArray<OBJC*>* innerArray = [NSMutableArray arrayWithCapacity:innerVector.size()];
        for (size_t index2 = 0; index2 < innerVector.size(); index2++) {
            [innerArray addObject:converter(innerVector[index2])];
        }
        [array addObject:innerArray];
    }
}

#define CV2OBJC2(CV_CLASS, OBJC_CLASS, v, a) \
    cv2objc2<CV_CLASS, OBJC_CLASS>(v, a, [](CV_CLASS& cv) -> OBJC_CLASS* { return [OBJC_CLASS fromNative:cv]; })

#endif
