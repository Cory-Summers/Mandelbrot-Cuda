#include "cudaImmediary.h"
#include "mandelbrot-kernel.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#define CudaErrorCheck(var, function) \
var = function; \
if(_CudaErrorCheck(error)) { return 1; }

bool _CudaErrorCheck(cudaError_t const& error)
{
  if (error != cudaSuccess)
  {
    fprintf(stderr, "%s>%s\n", cudaGetErrorName(error), cudaGetErrorString(error));
    return 1;
  }
  return 0;
}
int DivUp(int a, int b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}
int InitializeCudaData(cuMandelbrotData* cuData, const size_t buffer_size)
{
  cudaError_t error;
  CudaErrorCheck(error, cudaMalloc(&(cuData->cuBuffer), buffer_size));
  CudaErrorCheck(error, cudaMalloc(&(cuData->cuMandel), sizeof(MandelbrotPlot_t)));
  cuData->init = true;
  cuData->buffer_size = buffer_size;
  return 0;
}
int UpdateCudaData(cuMandelbrotData& cuData, MandelbrotPlot_t const& plot)
{
  cudaError_t error;
  const unsigned PLOT_SIZE = (plot.height * plot.width * 4);
  if ((cuData.buffer_size) != PLOT_SIZE)
  {
    CudaErrorCheck(error, cudaFree(cuData.cuBuffer));
    CudaErrorCheck(error, cudaMalloc(&(cuData.cuBuffer), PLOT_SIZE));
    cuData.buffer_size = PLOT_SIZE;
  }
  CudaErrorCheck(error, cudaMemcpy(cuData.cuMandel, &plot, sizeof(MandelbrotPlot_t), cudaMemcpyHostToDevice));
  return 0;
}
uint8_t* MandelbrotCudaCall(
  uint8_t * buffer,
  MandelbrotPlot_t const & plot,
  cuMandelbrotData & cuBuffers
)
{
  buffer[0] =0xD;
  if (cuBuffers.init != true)
    InitializeCudaData(&cuBuffers, plot.height * plot.width * 4);
  UpdateCudaData(cuBuffers, plot);
  dim3 bs(64, 4), grid(DivUp(plot.width, bs.x), DivUp(plot.height, bs.y));
  MandelbrotKernel <<<grid, bs >>> (cuBuffers.cuBuffer, cuBuffers.cuMandel);
  cudaMemcpy(buffer, cuBuffers.cuBuffer, cuBuffers.buffer_size, cudaMemcpyDeviceToHost);
  return NULL;
}
