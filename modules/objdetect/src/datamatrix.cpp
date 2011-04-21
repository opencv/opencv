#include <xmmintrin.h>
#include "precomp.hpp"
#include <deque>
using namespace std;

#undef NDEBUG
#include <assert.h>

class Sampler {
public:
  CvMat *im;
  CvPoint o;
  CvPoint c, cc;
  CvMat *perim;
  CvPoint fcoord(float fx, float fy);
  CvPoint coord(int ix, int iy);
  Sampler() {}
  Sampler(CvMat *_im, CvPoint _o, CvPoint _c, CvPoint _cc);
  uint8 getpixel(int ix, int iy);
  int isinside(int x, int y);
  int overlap(Sampler &other);
  int hasbars();
  void timing();
  CvMat *extract();
};

class code {    // used in this file only
public:
  char msg[4];
  CvMat *original;
  Sampler sa;
};

#include "followblk.h"

#define dethresh 0.92f
#define eincO    (2 * dethresh)         // e increment orthogonal
#define eincD    (1.414 * dethresh)     // e increment diagonal

static const float eincs[] = {
  eincO, eincD,
  eincO, eincD,
  eincO, eincD,
  eincO, eincD,
  999 };

#define Ki(x) _mm_set_epi32((x),(x),(x),(x))
#define Kf(x) _mm_set_ps((x),(x),(x),(x))
#define _mm_abs_ps(x) (__m128)_mm_and_ps((__m128)(x), (__m128)Ki(0x7fffffff))

static void writexy(CvMat *m, int r, CvPoint p)
{
  int *pdst = (int*)cvPtr2D(m, r, 0);
  pdst[0] = p.x;
  pdst[1] = p.y;
}

Sampler::Sampler(CvMat *_im, CvPoint _o, CvPoint _c, CvPoint _cc)
{
  im = _im;
  o = _o;
  c = _c;
  cc = _cc;
  perim = cvCreateMat(4, 1, CV_32SC2);
  writexy(perim, 0, fcoord(-.2,-.2));
  writexy(perim, 1, fcoord(-.2,1.2));
  writexy(perim, 2, fcoord(1.2,1.2));
  writexy(perim, 3, fcoord(1.2,-.2));
  // printf("Sampler %d,%d %d,%d %d,%d\n", o.x, o.y, c.x, c.y, cc.x, cc.y);
}

CvPoint Sampler::fcoord(float fx, float fy)
{
  CvPoint r;
  r.x = (int)(o.x + fx * (cc.x - o.x) + fy * (c.x - o.x));
  r.y = (int)(o.y + fx * (cc.y - o.y) + fy * (c.y - o.y));
  return r;
}

CvPoint Sampler::coord(int ix, int iy)
{
  return fcoord(0.05 + 0.1 * ix, 0.05 + 0.1 * iy);
}

uint8 Sampler::getpixel(int ix, int iy)
{
  CvPoint pt = coord(ix, iy);
  // printf("%d,%d\n", pt.x, pt.y);
  return *cvPtr2D(im, pt.y, pt.x);
}

int Sampler::isinside(int x, int y)
{
  CvPoint2D32f fp;
  fp.x = x;
  fp.y = y;
  return cvPointPolygonTest(perim, fp, 0) < 0;
}

int Sampler::overlap(Sampler &other)
{
  for (int i = 0; i < 4; i++) {
    CvScalar p;
    p = cvGet2D(other.perim, i, 0);
    if (isinside((int)p.val[0], (int)p.val[1]))
      return 1;
    p = cvGet2D(perim, i, 0);
    if (other.isinside((int)p.val[0], (int)p.val[1]))
      return 1;
  }
  return 0;
}

int Sampler::hasbars()
{
  return getpixel(9, 1) > getpixel(9, 0);
}

void Sampler::timing()
{
  uint8 dark = getpixel(9, 0);
  for (int i = 1; i < 3; i += 2) {
    uint8 light = getpixel(9, i);
    // if (light <= dark)
    //  goto endo;
    dark = getpixel(9, i + 1);
    // if (up <= down)
    //  goto endo;
  }
}

CvMat *Sampler::extract()
{
  // return a 10x10 CvMat for the current contents, 0 is black, 255 is white
  // Sampler has (0,0) at bottom left, so invert Y
  CvMat *r = cvCreateMat(10, 10, CV_8UC1);
  for (int x = 0; x < 10; x++)
    for (int y = 0; y < 10; y++)
      *cvPtr2D(r, 9 - y, x) = (getpixel(x, y) < 128) ? 0 : 255;
  return r;
}

static void apron(CvMat *v)
{
  int r = v->rows;
  int c = v->cols;
  memset(cvPtr2D(v, 0, 0), 0x22, c);
  memset(cvPtr2D(v, 1, 0), 0x22, c);
  memset(cvPtr2D(v, r - 2, 0), 0x22, c);
  memset(cvPtr2D(v, r - 1, 0), 0x22, c);
  int y;
  for (y = 2; y < r - 2; y++) {
    uchar *lp = cvPtr2D(v, y, 0);
    lp[0] = 0x22;
    lp[1] = 0x22;
    lp[c-2] = 0x22;
    lp[c-1] = 0x22;
  }
}

static void cfollow(CvMat *src, CvMat *dst)
{
  int sx, sy;
  uint8 *vpd = cvPtr2D(src, 0, 0);
  for (sy = 0; sy < src->rows; sy++) {
    short *wr = (short*)cvPtr2D(dst, sy, 0);
    for (sx = 0; sx < src->cols; sx++) {
      int x = sx;
      int y = sy;
      float e = 0;
      int ontrack = true;
      int dir;

      while (ontrack) {
        dir = vpd[y * src->step + x];
        int xd = ((dir & 0xf) - 2);
        int yd = ((dir >> 4) - 2);
        e += (dir == 0x22) ? 999 : ((dir & 1) ? eincD : eincO);
        x += xd;
        y += yd;
        if (e > 10.) {
          float d = ((x - sx) * (x - sx)) + ((y - sy) * (y - sy));
          ontrack = d > (e * e);
        }
      }
      if ((24 <= e) && (e < 999)) {
        // printf("sx=%d, sy=%d, x=%d, y=%d\n", sx, sy, x, y);
        *wr++ = x - sx;
        *wr++ = y - sy;
      } else {
        *wr++ = 0;
        *wr++ = 0;
      }
    }
  }
}

static uint8 gf256mul(uint8 a, uint8 b)
{
    return Alog[(Log[a] + Log[b]) % 255];
}

static int decode(Sampler &sa, code &cc)
{
  uint8 binary[8] = {0,0,0,0,0,0,0,0};
  uint8 b = 0;

  for (int i = 0; i < 64; i++) {
    b = (b << 1) + (sa.getpixel(pickup[i].x, pickup[i].y) <= 128);
    if ((i & 7) == 7) {
      binary[i >> 3] = b;
      b = 0;
    }
  }

  // Compute the 5 RS codewords for the 3 datawords

  uint8 c[5] = {0,0,0,0,0};
  {
    int i, j;
    uint8 a[5] = {228, 48, 15, 111, 62};
    int k = 5;
    for (i = 0; i < 3; i++) {
      uint8 t = binary[i] ^ c[4];
      for (j = k - 1; j != -1; j--) {
        if (t == 0)
            c[j] = 0;
        else
            c[j] = gf256mul(t, a[j]);
        if (j > 0)
            c[j] = c[j - 1] ^ c[j];
      }
    }
  }

  if ((c[4] == binary[3]) &&
      (c[3] == binary[4]) &&
      (c[2] == binary[5]) &&
      (c[1] == binary[6]) &&
      (c[0] == binary[7])) {
    uint8 x = 0xff & (binary[0] - 1);
    uint8 y = 0xff & (binary[1] - 1);
    uint8 z = 0xff & (binary[2] - 1);
    cc.msg[0] = x;
    cc.msg[1] = y;
    cc.msg[2] = z;
    cc.msg[3] = 0;
    cc.sa = sa;
    cc.original = sa.extract();
    return 1;
  } else {
    return 0;
  }
}

static deque<CvPoint> trailto(CvMat *v, int x, int y, CvMat *terminal)
{
  CvPoint np;
  /* Return the last 10th of the trail of points following v from (x,y)
   * to terminal
   */

  int ex = x + ((short*)cvPtr2D(terminal, y, x))[0];
  int ey = y + ((short*)cvPtr2D(terminal, y, x))[1];
  deque<CvPoint> r;
  while ((x != ex) || (y != ey)) {
    np.x = x;
    np.y = y;
    r.push_back(np);
    int dir = *cvPtr2D(v, y, x);
    int xd = ((dir & 0xf) - 2);
    int yd = ((dir >> 4) - 2);
    x += xd;
    y += yd;
  }

  int l = r.size() * 9 / 10;
  while (l--)
    r.pop_front();
  return r;
}

deque <DataMatrixCode> cvFindDataMatrix(CvMat *im)
{
  int r = im->rows;
  int c = im->cols;

#define SAMESIZE(nm, ty) CvMat *nm = cvCreateMat(r, c, ty);

  SAMESIZE(thresh, CV_8UC1)
  SAMESIZE(vecpic, CV_8UC1)
  SAMESIZE(vc, CV_8UC1)
  SAMESIZE(vcc, CV_8UC1)
  SAMESIZE(cxy, CV_16SC2)
  SAMESIZE(ccxy, CV_16SC2)

  cvAdaptiveThreshold(im, thresh, 255.0, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 13);
  {
    int x, y;
    int sstride = thresh->step;
    int sw = thresh->cols; // source width
    for (y = 2; y < thresh->rows - 2; y++) {
      uint8 *ps = cvPtr2D(thresh, y, 0);
      uint8 *pd = cvPtr2D(vecpic, y, 0);
      uint8 *pvc = cvPtr2D(vc, y, 0);
      uint8 *pvcc = cvPtr2D(vcc, y, 0);
      for (x = 0; x < sw; x++) {
        uint8 v =
            (0x01 & ps[-2 * sstride]) |
            (0x02 & ps[-sstride + 1]) |
            (0x04 & ps[2]) |
            (0x08 & ps[sstride + 1]) |
            (0x10 & ps[2 * sstride]) |
            (0x20 & ps[sstride - 1]) |
            (0x40 & ps[-2]) |
            (0x80 & ps[-sstride -1]);
        *pd++ = v;
        *pvc++ = cblk[v];
        *pvcc++ = ccblk[v];
        ps++;
      }
    }
    apron(vc);
    apron(vcc);
  }

  cfollow(vc, cxy);
  cfollow(vcc, ccxy);

  deque <CvPoint> candidates;
  {
    int x, y;
    int r = cxy->rows;
    int c = cxy->cols;
    for (y = 0; y < r; y++) {
      __m64 *cd = (__m64 *)cvPtr2D(cxy, y, 0);
      __m64 *ccd = (__m64 *)cvPtr2D(ccxy, y, 0);
      for (x = 0; x < c; x += 4) {
        __m128 cyxyxA = _mm_cvtpi16_ps(*cd++);
        __m128 cyxyxB = _mm_cvtpi16_ps(*cd++);
        __m128 cx = _mm_shuffle_ps(cyxyxA, cyxyxB, _MM_SHUFFLE(0, 2, 0, 2));
        __m128 cy = _mm_shuffle_ps(cyxyxA, cyxyxB, _MM_SHUFFLE(1, 3, 1, 3));
        __m128 cmag = _mm_sqrt_ps(cx * cx + cy * cy);
        __m128 crmag = _mm_rcp_ps(cmag);
        __m128 ncx = cx * crmag;
        __m128 ncy = cy * crmag;

        __m128 ccyxyxA = _mm_cvtpi16_ps(*ccd++);
        __m128 ccyxyxB = _mm_cvtpi16_ps(*ccd++);
        __m128 ccx = _mm_shuffle_ps(ccyxyxA, ccyxyxB, _MM_SHUFFLE(0, 2, 0, 2));
        __m128 ccy = _mm_shuffle_ps(ccyxyxA, ccyxyxB, _MM_SHUFFLE(1, 3, 1, 3));
        __m128 ccmag = _mm_sqrt_ps(ccx * ccx + ccy * ccy);
        __m128 ccrmag = _mm_rcp_ps(ccmag);
        __m128 nccx = ccx * ccrmag;
        __m128 nccy = ccy * ccrmag;

        __m128 dot = ncx * nccx + ncy * nccy;
        // iscand = (cmag > 30) & (ccmag > 30) & (numpy.minimum(cmag, ccmag) * 1.1 > numpy.maximum(cmag, ccmag)) & (abs(dot) < 0.25)
        __m128 iscand = _mm_and_ps(_mm_cmpgt_ps(cmag, Kf(30)), _mm_cmpgt_ps(ccmag, Kf(30)));
        iscand = _mm_and_ps(iscand, _mm_cmpgt_ps(_mm_min_ps(cmag, ccmag) * Kf(1.1), _mm_max_ps(cmag, ccmag)));
        iscand = _mm_and_ps(iscand, _mm_cmplt_ps(_mm_abs_ps(dot), Kf(0.25)));

        unsigned int result[4];
        *(__m128*)result = iscand;
        int ix;
        CvPoint np;
        for (ix = 0; ix < 4; ix++) {
          if (result[ix]) {
            np.x = x + ix;
            np.y = y;
            candidates.push_back(np);
          }
        }
      }
    }
  }

  deque <code> codes;
  size_t i, j, k;
  while (!candidates.empty()) {
    CvPoint o = candidates.front();
    candidates.pop_front();
    deque<CvPoint> ptc = trailto(vc, o.x, o.y, cxy);
    deque<CvPoint> ptcc = trailto(vcc, o.x, o.y, ccxy);
    for (j = 0; j < ptc.size(); j++) {
      for (k = 0; k < ptcc.size(); k++) {
        code cc;
        Sampler sa(im, o, ptc[j], ptcc[k]);
        for (i = 0; i < codes.size(); i++) {
          if (sa.overlap(codes[i].sa))
            goto endo;
        }
        if (codes.size() > 0) {
          printf("searching for more\n");
        }
        if (decode(sa, cc)) {
          codes.push_back(cc);
          goto endo;
        }
      }
    }
endo: ; // end search for this o
  }

  cvFree(&thresh);
  cvFree(&vecpic);
  cvFree(&vc);
  cvFree(&vcc);
  cvFree(&cxy);
  cvFree(&ccxy);

  deque <DataMatrixCode> rc;
  for (i = 0; i < codes.size(); i++) {
    DataMatrixCode cc;
    strcpy(cc.msg, codes[i].msg);
    cc.original = codes[i].original;
    cc.corners = codes[i].sa.perim;
    rc.push_back(cc);
  }
  return rc;
}
