// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef G2O_SIM_3
#define G2O_SIM_3

#include "g2o/types/slam3d/se3_ops.h"
#include "g2o/stuff/misc.h"
#include <Eigen/Geometry>

namespace g2o
{
  struct Sim3
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    protected:
      Quaternion r;
      Vector3 t;
      number_t s;

    public:
      Sim3()
      {
        r.setIdentity();
        t.fill(0.);
        s=1.;
      }

      Sim3(const Quaternion & r, const Vector3 & t, number_t s)
        : r(r),t(t),s(s)
      {
			normalizeRotation();
      }

      Sim3(const Matrix3 & R, const Vector3 & t, number_t s)
        : r(Quaternion(R)),t(t),s(s)
      {
			normalizeRotation();
      }


      Sim3(const Vector7 & update)
      {

        Vector3 omega;
        for (int i=0; i<3; i++)
          omega[i]=update[i];

        Vector3 upsilon;
        for (int i=0; i<3; i++)
          upsilon[i]=update[i+3];

        number_t sigma = update[6];
        number_t theta = omega.norm();
        Matrix3 Omega = skew(omega);
        s = std::exp(sigma);
        Matrix3 Omega2 = Omega*Omega;
        Matrix3 I;
        I.setIdentity();
        Matrix3 R;

        number_t eps = cst(0.00001);
        number_t A,B,C;
        if (fabs(sigma)<eps)
        {
          C = 1;
          if (theta<eps)
          {
            A = cst(1./2.);
            B = cst(1./6.);
            R = (I + Omega + Omega*Omega/2);//R=I+(1-cos(theta))*a^a^+sin(theta)*a^~=(omit O(theta^3))=I+theta^2/2*a^a^+theta*a^
          }
          else
          {
            number_t theta2= theta*theta;
            A = (1-std::cos(theta))/(theta2);
            B = (theta-std::sin(theta))/(theta2*theta);
            R = I + std::sin(theta)/theta *Omega + (1-std::cos(theta))/(theta*theta)*Omega2;
          }
        }
        else
        {
          C=(s-1)/sigma;
          if (theta<eps)
          {
            number_t sigma2= sigma*sigma;
            A = ((sigma-1)*s+1)/sigma2;
            B= ((cst(0.5)*sigma2-sigma+1)*s-1)/(sigma2*sigma);//B=[C-((s*cos(theta)-1)*sigma+s*sin(theta)*theta)/(sigma^2+theta^2)]/theta^2~=(omit O(theta^2))=
	    //(1/2*s*sigma-s)/(sigma^2)+[C-(s-1)*sigma/(sigma^2+theta^2)]/theta^2~=(0.5*sigma^2*s-s*sigma)/sigma^3+[s-1]/sigma^3=[s*(0.5*sigma^2-sigma+1)-1]/sigma^3  
            R = (I + Omega + Omega2/2);//R=I+(1-cos(theta))*a^a^+sin(theta)*a^~=I+theta^2/2*a^a^+theta*a^
          }
          else
          {
            R = I + std::sin(theta)/theta *Omega + (1-std::cos(theta))/(theta*theta)*Omega2;
            number_t a=s*std::sin(theta);
            number_t b=s*std::cos(theta);
            number_t theta2= theta*theta;
            number_t sigma2= sigma*sigma;
            number_t c=theta2+sigma2;
            A = (a*sigma+ (1-b)*theta)/(theta*c);
            B = (C-((b-1)*sigma+a*theta)/(c))*1/(theta2);

          }
        }
        r = Quaternion(R);



        Matrix3 W = A*Omega + B*Omega2 + C*I;
        t = W*upsilon;
      }

      Vector3 map (const Vector3& xyz) const {
        return s*(r*xyz) + t;
      }

      Vector7 log() const
      {
        Vector7 res;
        number_t sigma = std::log(s);

        Vector3 omega;
        Vector3 upsilon;

        Matrix3 R = r.toRotationMatrix();
        number_t d =  cst(0.5)*(R(0,0)+R(1,1)+R(2,2)-1);

        Matrix3 Omega;

        number_t eps = cst(0.00001);
        Matrix3 I = Matrix3::Identity();

        number_t A,B,C;
        if (fabs(sigma)<eps)
        {
          C = 1;
          if (d>1-eps)
          {
            omega=0.5*deltaR(R);
            Omega = skew(omega);
            A = cst(1./2.);
            B = cst(1./6.);
          }
          else
          {
            number_t theta = std::acos(d);
            number_t theta2 = theta*theta;
            omega = theta/(2*std::sqrt(1-d*d))*deltaR(R);
            Omega = skew(omega);
            A = (1-std::cos(theta))/(theta2);
            B = (theta-std::sin(theta))/(theta2*theta);
          }
        }
        else
        {
          C=(s-1)/sigma;
          if (d>1-eps)
          {
            number_t sigma2 = sigma*sigma;
            omega=cst(0.5)*deltaR(R);
            Omega = skew(omega);
            A = ((sigma-1)*s+1)/(sigma2);
            B = ((cst(0.5)*sigma2-sigma+1)*s-1)/(sigma2*sigma);//B=[C-((s*cos(theta)-1)*sigma+s*sin(theta)*theta)/(sigma^2+theta^2)]/theta^2
	    //use limit(theta->0)(B)=limit(theta->0){[(sigma2+theta2)*(s*sigma*sin(theta)-s*sin(theta)-s*theta*cos(theta))+(s*cos(theta)*sigma-sigma+s*sin(theta)*theta)*2*theta]/(2*theta)}=
	    //=limit(theta->0)(s*sigma-s)*sin(theta)/(2*(sigma2+theta2)*theta)+limit(theta->0)[-s*cos(theta)/(2*(sigma2+theta2))+(s*cos(theta)*sigma-sigma+s*sin(theta)*theta)/(sigma2+theta2)^2]=
	    //=limit(theta->0)(s*sigma-s)*cos(theta)/(2*(sigma2+3*theta2))+-s/(2*sigma2)+(s-1)/sigma^3=
	    //=(s*sigma-s)/2/sigma2-s/2/sigma2+(s-1)/sigma^3=[(0.5*sigma2-sigma+1)*s-1]/sigma^3
          }
          else
          {
            number_t theta = std::acos(d);
            omega = theta/(2*std::sqrt(1-d*d))*deltaR(R);
            Omega = skew(omega);
            number_t theta2 = theta*theta;
            number_t a=s*std::sin(theta);
            number_t b=s*std::cos(theta);
            number_t c=theta2 + sigma*sigma;
            A = (a*sigma+ (1-b)*theta)/(theta*c);
            B = (C-((b-1)*sigma+a*theta)/(c))*1/(theta2);
          }
        }

        Matrix3 W = A*Omega + B*Omega*Omega + C*I;

        upsilon = W.lu().solve(t);

        for (int i=0; i<3; i++)
          res[i] = omega[i];

        for (int i=0; i<3; i++)
          res[i+3] = upsilon[i];

        res[6] = sigma;

        return res;
      }


      Sim3 inverse() const
      {
        return Sim3(r.conjugate(), r.conjugate()*((-1/s)*t), 1/s);
      }

      number_t operator[](int i) const
      {
        assert(i<8);
        if (i<4){

          return r.coeffs()[i];
        }
        if (i<7){
          return t[i-4];
        }
        return s;
      }

      number_t& operator[](int i)
      {
        assert(i<8);
        if (i<4){

          return r.coeffs()[i];
        }
        if (i<7)
        {
          return t[i-4];
        }
        return s;
      }

      Sim3 operator *(const Sim3& other) const {
        Sim3 ret;
        ret.r = r*other.r;
        ret.t=s*(r*other.t)+t;
        ret.s=s*other.s;
        return ret;
      }

      Sim3& operator *=(const Sim3& other){
        Sim3 ret=(*this)*other;
        *this=ret;
        return *this;
      }
    void normalizeRotation(){
        if (r.w()<0){
          r.coeffs() *= -1;
        }
        r.normalize();
    }
      inline const Vector3& translation() const {return t;}

      inline Vector3& translation() {return t;}

      inline const Quaternion& rotation() const {return r;}

      inline Quaternion& rotation() {return r;}

      inline const number_t& scale() const {return s;}

      inline number_t& scale() {return s;}

  };

  inline std::ostream& operator <<(std::ostream& out_str,
                                   const Sim3& sim3)
  {
    out_str << sim3.rotation().coeffs() << std::endl;
    out_str << sim3.translation() << std::endl;
    out_str << sim3.scale() << std::endl;

    return out_str;
  }

} // end namespace


#endif
