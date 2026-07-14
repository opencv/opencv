//
//  Moments.mm
//
//  Created by Giles Payne on 2019/10/09.
//

#import "Moments.h"

@implementation Moments {
    cv::Moments native;
}

-(cv::Moments&)nativeRef {
    return native;
}

- (double)m00 {
    return native.m00;
}

- (void)setM00:(double)val {
    native.m00 = val;
}

- (double)m10 {
    return native.m10;
}

- (void)setM10:(double)val {
    native.m10 = val;
}

- (double)m01 {
    return native.m01;
}

- (void)setM01:(double)val {
    native.m01 = val;
}

- (double)m20 {
    return native.m20;
}

- (void)setM20:(double)val {
    native.m20 = val;
}

- (double)m11 {
    return native.m11;
}

- (void)setM11:(double)val {
    native.m11 = val;
}

- (double)m02 {
    return native.m02;
}

- (void)setM02:(double)val {
    native.m02 = val;
}

- (double)m30 {
    return native.m30;
}

- (void)setM30:(double)val {
    native.m30 = val;
}

- (double)m21 {
    return native.m21;
}

- (void)setM21:(double)val {
    native.m21 = val;
}

- (double)m12 {
    return native.m12;
}

- (void)setM12:(double)val {
    native.m12 = val;
}

- (double)m03 {
    return native.m03;
}

- (void)setM03:(double)val {
    native.m03 = val;
}

- (double)mu20 {
    return native.mu20;
}

- (void)setMu20:(double)val {
    native.mu20 = val;
}

- (double)mu11 {
    return native.mu11;
}

- (void)setMu11:(double)val {
    native.mu11 = val;
}

- (double)mu02 {
    return native.mu02;
}

- (void)setMu02:(double)val {
    native.mu02 = val;
}

- (double)mu30 {
    return native.mu30;
}

- (void)setMu30:(double)val {
    native.mu30 = val;
}

- (double)mu21 {
    return native.mu21;
}

- (void)setMu21:(double)val {
    native.mu21 = val;
}
- (double)mu12 {
    return native.mu12;
}

- (void)setMu12:(double)val {
    native.mu12 = val;
}

- (double)mu03 {
    return native.mu03;
}

- (void)setMu03:(double)val {
    native.mu03 = val;
}

- (double)nu20 {
    return native.nu20;
}

- (void)setNu20:(double)val {
    native.nu20 = val;
}

- (double)nu11 {
    return native.nu11;
}

- (void)setNu11:(double)val {
    native.nu11 = val;
}

- (double)nu02 {
    return native.nu02;
}

- (void)setNu02:(double)val {
    native.nu02 = val;
}

- (double)nu30 {
    return native.nu30;
}

- (void)setNu30:(double)val {
    native.nu30 = val;
}

- (double)nu21 {
    return native.nu21;
}

- (void)setNu21:(double)val {
    native.nu21 = val;
}

- (double)nu12 {
    return native.nu12;
}

- (void)setNu12:(double)val {
    native.nu12 = val;
}

- (double)nu03 {
    return native.nu03;
}

- (void)setNu03:(double)val {
    native.nu03 = val;
}

-(instancetype)initWithM00:(double)m00 m10:(double)m10 m01:(double)m01 m20:(double)m20 m11:(double)m11 m02:(double)m02 m30:(double)m30 m21:(double)m21 m12:(double)m12 m03:(double)m03 {
    self = [super init];
    if (self) {
        self.m00 = m00;
        self.m10 = m10;
        self.m01 = m01;
        self.m20 = m20;
        self.m11 = m11;
        self.m02 = m02;
        self.m30 = m30;
        self.m21 = m21;
        self.m12 = m12;
        self.m03 = m03;
        [self completeState];
    }
    return self;
}
-(instancetype)init {
    return [self initWithM00:0 m10:0 m01:0 m20:0 m11:0 m02:0 m30:0 m21:0 m12:0 m03:0];
}

-(instancetype)initWithVals:(NSArray<NSNumber*>*)vals {
    self = [super init];
    if (self) {
        [self set:vals];
    }
    return self;
}

+(instancetype)fromNative:(cv::Moments&)moments {
    return [[Moments alloc] initWithM00:moments.m00 m10:moments.m10 m01:moments.m01 m20:moments.m20 m11:moments.m11 m02:moments.m02 m30:moments.m30 m21:moments.m21 m12:moments.m12 m03:moments.m03];
}

-(void)set:(NSArray<NSNumber*>*)vals {
    self.m00 = (vals != nil && vals.count > 0) ? vals[0].doubleValue : 0;
    self.m10 = (vals != nil && vals.count > 1) ? vals[1].doubleValue : 0;
    self.m01 = (vals != nil && vals.count > 2) ? vals[2].doubleValue : 0;
    self.m20 = (vals != nil && vals.count > 3) ? vals[3].doubleValue : 0;
    self.m11 = (vals != nil && vals.count > 4) ? vals[4].doubleValue : 0;
    self.m02 = (vals != nil && vals.count > 5) ? vals[5].doubleValue : 0;
    self.m30 = (vals != nil && vals.count > 6) ? vals[6].doubleValue : 0;
    self.m21 = (vals != nil && vals.count > 7) ? vals[7].doubleValue : 0;
    self.m12 = (vals != nil && vals.count > 8) ? vals[8].doubleValue : 0;
    self.m03 = (vals != nil && vals.count > 9) ? vals[9].doubleValue : 0;
    [self completeState];
}

-(void)completeState {
    double cx = 0, cy = 0;
    double mu20, mu11, mu02;
    double inv_m00 = 0.0;

    if (abs(self.m00) > 0.00000001) {
        inv_m00 = 1. / self.m00;
        cx = self.m10 * inv_m00;
        cy = self.m01 * inv_m00;
    }

    // mu20 = m20 - m10*cx
    mu20 = self.m20 - self.m10 * cx;
    // mu11 = m11 - m10*cy
    mu11 = self.m11 - self.m10 * cy;
    // mu02 = m02 - m01*cy
    mu02 = self.m02 - self.m01 * cy;

    self.mu20 = mu20;
    self.mu11 = mu11;
    self.mu02 = mu02;

    // mu30 = m30 - cx*(3*mu20 + cx*m10)
    self.mu30 = self.m30 - cx * (3 * mu20 + cx * self.m10);
    mu11 += mu11;
    // mu21 = m21 - cx*(2*mu11 + cx*m01) - cy*mu20
    self.mu21 = self.m21 - cx * (mu11 + cx * self.m01) - cy * mu20;
    // mu12 = m12 - cy*(2*mu11 + cy*m10) - cx*mu02
    self.mu12 = self.m12 - cy * (mu11 + cy * self.m10) - cx * mu02;
    // mu03 = m03 - cy*(3*mu02 + cy*m01)
    self.mu03 = self.m03 - cy * (3 * mu02 + cy * self.m01);


    double inv_sqrt_m00 = sqrt(abs(inv_m00));
    double s2 = inv_m00*inv_m00, s3 = s2*inv_sqrt_m00;

    self.nu20 = self.mu20*s2;
    self.nu11 = self.mu11*s2;
    self.nu02 = self.mu02*s2;
    self.nu30 = self.mu30*s3;
    self.nu21 = self.mu21*s3;
    self.nu12 = self.mu12*s3;
    self.nu03 = self.mu03*s3;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"Moments [ \nm00=%lf, \nm10=%lf, m01=%lf, \nm20=%lf, m11=%lf, m02=%lf, \nm30=%lf, m21=%lf, m12=%lf, m03=%lf, \nmu20=%lf, mu11=%lf, mu02=%lf, \nmu30=%lf, mu21=%lf, mu12=%lf, mu03=%lf, \nnu20=%lf, nu11=%lf, nu02=%lf, \nnu30=%lf, nu21=%lf, nu12=%lf, nu03=%lf, \n]", self.m00, self.m10, self.m01, self.m20, self.m11, self.m02, self.m30, self.m21, self.m12, self.m03, self.mu20, self.mu11, self.mu02, self.mu30, self.mu21, self.mu12, self.mu03, self.nu20, self.nu11, self.nu02, self.nu30, self.nu21, self.nu12, self.nu03];
}

@end
