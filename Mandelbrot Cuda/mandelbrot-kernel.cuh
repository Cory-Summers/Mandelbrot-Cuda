#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include <stdint.h>
#include "mandelbrot-structs.hpp"
int __host__ __device__ PixelDwell(double, double , double, double, int x, int y, int w, int h);

struct RGB
{
  uint8_t r;
  uint8_t g;
  uint8_t b;
};
__device__ float HuetoRGBA(float p, float q, float t);
__device__ RGB HSLtoRGBA(float h, float s, float l);
__global__ void MandelbrotKernel(uint8_t*, MandelbrotPlot_t * dat);
