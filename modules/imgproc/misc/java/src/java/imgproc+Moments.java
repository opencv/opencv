package org.opencv.imgproc;

import java.lang.Math;

//javadoc:Moments
public class Moments {

    public double m00;
    public double m10;
    public double m01;
    public double m20;
    public double m11;
    public double m02;
    public double m30;
    public double m21;
    public double m12;
    public double m03;

    public double mu20;
    public double mu11;
    public double mu02;
    public double mu30;
    public double mu21;
    public double mu12;
    public double mu03;

    public double nu20;
    public double nu11;
    public double nu02;
    public double nu30;
    public double nu21;
    public double nu12;
    public double nu03;

    public Moments(
        double m00,
        double m10,
        double m01,
        double m20,
        double m11,
        double m02,
        double m30,
        double m21,
        double m12,
        double m03)
    {
        this.m00 = m00;
        this.m10 = m10;
        this.m01 = m01;
        this.m20 = m20;
        this.m11 = m11;
        this.m02 = m02;
        this.m30 = m30;
        this.m21 = m21;
        this.m12 = m12;
        this.m03 = m03;
        this.completeState();
    }

    public Moments() {
        this(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }

    public Moments(double[] vals) {
        set(vals);
    }

    public void set(double[] vals) {
        if (vals != null) {
            m00 = vals.length > 0 ? vals[0] : 0;
            m10 = vals.length > 1 ? vals[1] : 0;
            m01 = vals.length > 2 ? vals[2] : 0;
            m20 = vals.length > 3 ? vals[3] : 0;
            m11 = vals.length > 4 ? vals[4] : 0;
            m02 = vals.length > 5 ? vals[5] : 0;
            m30 = vals.length > 6 ? vals[6] : 0;
            m21 = vals.length > 7 ? vals[7] : 0;
            m12 = vals.length > 8 ? vals[8] : 0;
            m03 = vals.length > 9 ? vals[9] : 0;
            this.completeState();
        } else {
            m00 = 0;
            m10 = 0;
            m01 = 0;
            m20 = 0;
            m11 = 0;
            m02 = 0;
            m30 = 0;
            m21 = 0;
            m12 = 0;
            m03 = 0;
            mu20 = 0;
            mu11 = 0;
            mu02 = 0;
            mu30 = 0;
            mu21 = 0;
            mu12 = 0;
            mu03 = 0;
            nu20 = 0;
            nu11 = 0;
            nu02 = 0;
            nu30 = 0;
            nu21 = 0;
            nu12 = 0;
            nu03 = 0;
        }
    }

    @Override
    public String toString() {
        return "Moments [ " +
            "\n" +
            "m00=" + m00 + ", " +
            "\n" +
            "m10=" + m10 + ", " +
            "m01=" + m01 + ", " +
            "\n" +
            "m20=" + m20 + ", " +
            "m11=" + m11 + ", " +
            "m02=" + m02 + ", " +
            "\n" +
            "m30=" + m30 + ", " +
            "m21=" + m21 + ", " +
            "m12=" + m12 + ", " +
            "m03=" + m03 + ", " +
            "\n" +
            "mu20=" + mu20 + ", " +
            "mu11=" + mu11 + ", " +
            "mu02=" + mu02 + ", " +
            "\n" +
            "mu30=" + mu30 + ", " +
            "mu21=" + mu21 + ", " +
            "mu12=" + mu12 + ", " +
            "mu03=" + mu03 + ", " +
            "\n" +
            "nu20=" + nu20 + ", " +
            "nu11=" + nu11 + ", " +
            "nu02=" + nu02 + ", " +
            "\n" +
            "nu30=" + nu30 + ", " +
            "nu21=" + nu21 + ", " +
            "nu12=" + nu12 + ", " +
            "nu03=" + nu03 + ", " +
            "\n]";
    }

    protected void completeState()
    {
        double cx = 0, cy = 0;
        double mu20, mu11, mu02;
        double inv_m00 = 0.0;

        if( Math.abs(this.m00) > 0.00000001 )
        {
            inv_m00 = 1. / this.m00;
            cx = this.m10 * inv_m00;
            cy = this.m01 * inv_m00;
        }

        // mu20 = m20 - m10*cx
        mu20 = this.m20 - this.m10 * cx;
        // mu11 = m11 - m10*cy
        mu11 = this.m11 - this.m10 * cy;
        // mu02 = m02 - m01*cy
        mu02 = this.m02 - this.m01 * cy;

        this.mu20 = mu20;
        this.mu11 = mu11;
        this.mu02 = mu02;

        // mu30 = m30 - cx*(3*mu20 + cx*m10)
        this.mu30 = this.m30 - cx * (3 * mu20 + cx * this.m10);
        mu11 += mu11;
        // mu21 = m21 - cx*(2*mu11 + cx*m01) - cy*mu20
        this.mu21 = this.m21 - cx * (mu11 + cx * this.m01) - cy * mu20;
        // mu12 = m12 - cy*(2*mu11 + cy*m10) - cx*mu02
        this.mu12 = this.m12 - cy * (mu11 + cy * this.m10) - cx * mu02;
        // mu03 = m03 - cy*(3*mu02 + cy*m01)
        this.mu03 = this.m03 - cy * (3 * mu02 + cy * this.m01);


        double inv_sqrt_m00 = Math.sqrt(Math.abs(inv_m00));
        double s2 = inv_m00*inv_m00, s3 = s2*inv_sqrt_m00;

        this.nu20 = this.mu20*s2;
        this.nu11 = this.mu11*s2;
        this.nu02 = this.mu02*s2;
        this.nu30 = this.mu30*s3;
        this.nu21 = this.mu21*s3;
        this.nu12 = this.mu12*s3;
        this.nu03 = this.mu03*s3;

    }

    public double get_m00() { return this.m00; }
    public double get_m10() { return this.m10; }
    public double get_m01() { return this.m01; }
    public double get_m20() { return this.m20; }
    public double get_m11() { return this.m11; }
    public double get_m02() { return this.m02; }
    public double get_m30() { return this.m30; }
    public double get_m21() { return this.m21; }
    public double get_m12() { return this.m12; }
    public double get_m03() { return this.m03; }
    public double get_mu20() { return this.mu20; }
    public double get_mu11() { return this.mu11; }
    public double get_mu02() { return this.mu02; }
    public double get_mu30() { return this.mu30; }
    public double get_mu21() { return this.mu21; }
    public double get_mu12() { return this.mu12; }
    public double get_mu03() { return this.mu03; }
    public double get_nu20() { return this.nu20; }
    public double get_nu11() { return this.nu11; }
    public double get_nu02() { return this.nu02; }
    public double get_nu30() { return this.nu30; }
    public double get_nu21() { return this.nu21; }
    public double get_nu12() { return this.nu12; }
    public double get_nu03() { return this.nu03; }

    public void set_m00(double m00) { this.m00 = m00; }
    public void set_m10(double m10) { this.m10 = m10; }
    public void set_m01(double m01) { this.m01 = m01; }
    public void set_m20(double m20) { this.m20 = m20; }
    public void set_m11(double m11) { this.m11 = m11; }
    public void set_m02(double m02) { this.m02 = m02; }
    public void set_m30(double m30) { this.m30 = m30; }
    public void set_m21(double m21) { this.m21 = m21; }
    public void set_m12(double m12) { this.m12 = m12; }
    public void set_m03(double m03) { this.m03 = m03; }
    public void set_mu20(double mu20) { this.mu20 = mu20; }
    public void set_mu11(double mu11) { this.mu11 = mu11; }
    public void set_mu02(double mu02) { this.mu02 = mu02; }
    public void set_mu30(double mu30) { this.mu30 = mu30; }
    public void set_mu21(double mu21) { this.mu21 = mu21; }
    public void set_mu12(double mu12) { this.mu12 = mu12; }
    public void set_mu03(double mu03) { this.mu03 = mu03; }
    public void set_nu20(double nu20) { this.nu20 = nu20; }
    public void set_nu11(double nu11) { this.nu11 = nu11; }
    public void set_nu02(double nu02) { this.nu02 = nu02; }
    public void set_nu30(double nu30) { this.nu30 = nu30; }
    public void set_nu21(double nu21) { this.nu21 = nu21; }
    public void set_nu12(double nu12) { this.nu12 = nu12; }
    public void set_nu03(double nu03) { this.nu03 = nu03; }
}
