#pragma once
#ifdef __cplusplus
#include <cstdint>
#include <array>
#include <ostream>
#else
#include <stdint.h>
#endif
#include "mandelbrot-structs.hpp"
uint8_t* MandelbrotCudaCall(
  uint8_t * buffer,
  MandelbrotPlot_t const & plot,
  cuMandelbrotData &);
//Does not change data of plot. Only creates gpu memory.
int InitializeCudaData(cuMandelbrotData* cuData, const size_t buffer_size);
int UpdateCudaData(cuMandelbrotData&, MandelbrotPlot_t const&);
