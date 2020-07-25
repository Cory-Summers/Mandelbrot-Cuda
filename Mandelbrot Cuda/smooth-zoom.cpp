#include "smooth-zoom.hpp"
#include "mUtilities.h"
#include <chrono>
SmoothZoom::SmoothZoom()
  : m_buffers()
  , n_zooms(1)
  , max_zoom()
  , min_zoom()
  , init(false)
{
}

SmoothZoom::SmoothZoom(MandelbrotPlot_t const& min, MandelbrotPlot_t const& max, std::size_t size)
  : m_buffers(size)
  , n_zooms(size)
  , max_zoom(max)
  , min_zoom(min)
  , init(true)
{
  Initialize();
}

void SmoothZoom::Initialize(MandelbrotPlot_t const& min, MandelbrotPlot_t const& max, std::size_t size)
{
  n_zooms = size;
  max_zoom = max;
  min_zoom = min;
  init = true;
  Initialize();
}

void SmoothZoom::Render(MandelbrotSet & renderer)
{
  MandelbrotPlot_t frame = min_zoom;
  double dxl = (max_zoom.x_min - min_zoom.x_min) / n_zooms;
  double dxr = (max_zoom.x_max - min_zoom.x_max) / n_zooms;
  double dyt = (max_zoom.y_top - min_zoom.y_top) / n_zooms;
  double dyb = (max_zoom.y_bot - min_zoom.y_bot) / n_zooms;
  std::cout << dxl << ' ' << dxr << ' ' << dyt << ' ' << dyb << "\n";
  auto start = std::chrono::steady_clock::now();
  for(uint8_t * b : m_buffers)
  {
    renderer.Render(frame, b);
    frame.x_min += dxl;
    frame.x_max += dxr;
    frame.y_top += dyt;
    frame.y_bot += dyb;
  }
  auto end = std::chrono::steady_clock::now();
  std::cout << "Time to generate " << " = " << std::chrono::duration <double, std::milli>(end - start).count() << " [ms]" << std::endl;
  zoom_iter = m_buffers.begin();
  renderer.Update(frame);
}

SmoothZoom::~SmoothZoom()
{
  ClearBuffers();
}

 void SmoothZoom::Initialize()
{
   uint8_t* temp;
   buffer_size = min_zoom.BufferSize();
   if (buffer_size != max_zoom.BufferSize())
     throw;
   for (std::size_t i = 0; i < n_zooms; ++i) {
     temp = new uint8_t[buffer_size]();
     m_buffers.push_back(temp);
     if (m_buffers.back() == 0)
       throw "fuck";
   }
}
