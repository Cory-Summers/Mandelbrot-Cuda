#pragma once

#ifdef __cplusplus
#include <cstdint>
#include <array>
#include <ostream>
#else
#include <stdint.h>
#endif
struct MandelbrotPlot_t
{
  int width;
  int height;
  double x_min;
  double x_max;
  double y_top;
  double y_bot;
#ifdef __cplusplus
  MandelbrotPlot_t(int w, int h, double xm, double xx, double ym, double yx)
    : width(w)
    , height(h)
    , x_min(xm)
    , x_max(xx)
    , y_top(ym)
    , y_bot(yx)
  {}
  MandelbrotPlot_t()
    : width()
    , height()
    , x_min()
    , x_max()
    , y_top()
    , y_bot() {}
  std::array<double, 2> GetCenter() const
  {
    return std::array<double, 2>({ (x_max - x_min) / 2.0 + x_min, (y_bot - y_top) / 2.0 + y_top });
  }
  std::array<double, 2> GetDimensions() const
  {
    return std::array<double, 2>({ (x_max - x_min), (y_bot - y_top) });
  }
  std::size_t BufferSize() const { return height * width * 4; }
#endif
};
#ifdef __cplusplus
std::ostream& operator<<(std::ostream& os, MandelbrotPlot_t const& mp);
#endif
struct cuMandelbrotData
{
  uint8_t* cuBuffer;
  size_t   buffer_size;
  MandelbrotPlot_t* cuMandel;
  bool init;
};