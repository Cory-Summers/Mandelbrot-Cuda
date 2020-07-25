#include "mandelbrot-kernel.cuh"
#include <stdio.h>
#include <cuComplex.h>
#define MAX_DWELL 512
// w, h --- width and hight of the image, in pixels
// cmin, cmax --- coordinates of bottom-left and top-right image corners
// x, y --- coordinates of the pixel
typedef double     rtype;
typedef cuDoubleComplex ctype;
#define rpart(x)   (cuCreal(x))
#define ipart(x)   (cuCimag(x))
#define cmplx(x,y) (make_cuDoubleComplex(x,y))

__host__ __device__ rtype carg(const ctype& z) { return (rtype)atan2(ipart(z), rpart(z)); } // polar angle
__host__ __device__ rtype cabs(const ctype& z) { return (rtype)cuCabs(z); }
__host__ __device__ ctype cp2c(const rtype d, const rtype a) { return cmplx(d * cos(a), d * sin(a)); }
__host__ __device__ ctype cpow(const ctype& z, const int& n) { return cmplx((pow(cabs(z), n) * cos(n * carg(z))), (pow(cabs(z), n) * sin(n * carg(z)))); }
__device__ RGB get_rgb_smooth(int n, int iter_max) {
  // map n on the 0..1 interval
  RGB rgb;
  double t = (double)n / (double)iter_max;

  // Use smooth polynomials for r, g, b
  rgb.r = (int)(9 * (1 - t) * t * t * t * 255);
  rgb.g = (int)(15 * (1 - t) * (1 - t) * t * t * 255);
  rgb.b = (int)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
  return rgb;
}

__device__ RGB get_rgb_piecewise_linear(int n, int iter_max) {
  RGB rgb;
  int N = 256; // colors per element
  int N3 = N * N * N;
  // map n on the 0..1 interval (real numbers)
  double t = (double)n / (double)iter_max;
  // expand n on the 0 .. 256^3 interval (integers)
  n = (int)(t * (double)N3);

  int b = n / (N * N);
  int nn = n - b * N * N;
  int r = nn / N;
  int g = nn - r * N;
  rgb = { r, g, b};
  return rgb;
}
__device__ float HuetoRGBA(float p, float q, float t)
{
  if (t < 0.f)
    t += 1.f;
  if (t > 1.0f)
    t -= 1.f;
  if (t < 1.f / 6.f)
    return (p + (q - p) * 6.f * t);
  if (t < .5f)
    return q;
  if (t < 2.f / 3.f)
    return (p + (q - p) * (2.f / 3.f - t) * 6.f);
  return p;
}
RGB __device__ HSLtoRGBA(float h, float s, float l)
{
  RGB rgb;
  if (s == 0.f)
  {
    rgb.r = rgb.g = rgb.b = static_cast<uint8_t>(fminf(255.f, l * 256));
    return rgb;
  }
  float q = l < 0.5f ? l * (1 + s) : l + s - l * s;
  float p = 2 * l - q;
  rgb.r = static_cast<uint8_t>(fminf(255.f, HuetoRGBA(p, q, h + 1.f / 3.f) * 256.0f));
  rgb.g = static_cast<uint8_t>(fminf(255.f, HuetoRGBA(p, q, h) * 256.0f));
  rgb.b = static_cast<uint8_t>(fminf(255.f, HuetoRGBA(p, q, h - 1.f / 3.f) * 256.0f));
  return rgb;
}
int __device__ PixelDwell(double x_min, double x_max, double y_top, double y_bot, int x, int y, int w, int h) {
  cuDoubleComplex c = make_cuDoubleComplex(x, y);
  c = make_cuDoubleComplex(c.x / (double)w * (x_max - x_min) + x_min,
    c.y / (double)h * (y_bot - y_top) + y_top);
  cuDoubleComplex z = make_cuDoubleComplex(c.x, c.y);
  size_t iter = 0;
  while (iter < MAX_DWELL && cuCabs(z) < 2.0) {
    z = cuCadd(cpow(z, 2), c);
    ++iter;
  }
  return iter;
}

__global__ void MandelbrotKernel(uint8_t* buffer, MandelbrotPlot_t * p)
{
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int dwells;
  RGB color;
  int reverse = (p->height - 1);
  if (x < p->width && y < p->height) {
    dwells = PixelDwell(p->x_min, p->x_max, p->y_top, p->y_bot, x, y, p->width, p->height);
    //color = get_rgb_piecewise_linear(dwells, MAX_DWELL);
    color = get_rgb_smooth(dwells, MAX_DWELL);
    *(buffer + (x + (reverse - y) * p->width) * 4)     = color.r;
    *(buffer + (x + (reverse - y) * p->width) * 4 + 1) = color.g;
    *(buffer + (x + (reverse - y) * p->width) * 4 + 2) = color.b;
    *(buffer + (x + (reverse - y) * p->width) * 4 + 3) = 0xff;
  }
}
