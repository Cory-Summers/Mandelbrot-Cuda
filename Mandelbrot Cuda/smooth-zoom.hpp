#pragma once
#include <memory>
#include <iostream>
#include <vector>
#include <list>
#include "mandelbrot-structs.hpp"
#include "mandelbrot-class.hpp"
class SmoothZoom
{
public:
  SmoothZoom();
  SmoothZoom(MandelbrotPlot_t const &, MandelbrotPlot_t const &, std::size_t = 32ull);
  void Initialize(MandelbrotPlot_t const&, MandelbrotPlot_t const&, std::size_t = 32ull);
  void Render(MandelbrotSet &);
  void ClearBuffers() { for (auto b : m_buffers) delete[] b; }
  std::list<std::uint8_t*>::const_iterator& Iterator() {
    return zoom_iter;
  }
  std::list<std::uint8_t*> const& GetBuffers() const { return m_buffers; }
  ~SmoothZoom();
private:
  void Initialize();
  std::list<std::uint8_t*> m_buffers;
  std::size_t buffer_size;
  std::size_t n_zooms;
  MandelbrotPlot_t max_zoom; //smallest
  MandelbrotPlot_t min_zoom; //farthest
  std::list<std::uint8_t*>::const_iterator zoom_iter;
  bool init;
};

