#include "mandelbrot-structs.hpp"

std::ostream& operator<<(std::ostream& os, MandelbrotPlot_t const& mp)
{
  os << "{ " << mp.width << ", " << mp.height << ", " << mp.x_min
    << ", " << mp.x_max << ", " << mp.y_top << ", " << mp.y_bot << "}";
  return os;
}
