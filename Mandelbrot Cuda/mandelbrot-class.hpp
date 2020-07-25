#pragma once
#include "cudaImmediary.h"
#include <array>
#include <memory>
constexpr std::array<double, 4> MAX_DIMENSIONS = { -2.0, 1.0, 1.0, -1.0 };

constexpr std::size_t W_PIXEL = 4;
class MandelbrotSet
{
public:
  MandelbrotSet();
  MandelbrotSet(std::size_t const &, std::size_t const &);
  uint8_t const * Render();
  void Render(MandelbrotPlot_t const&, uint8_t* ext_buffer);
  inline uint8_t const* GetBuffer()  const { return m_buffer; }
  inline std::size_t GetBufferSize() const { return m_buffer_size; }
  float GetAspect() const { return (m_plot.height / static_cast<float>(m_plot.width)); }
  MandelbrotPlot_t const& GetPlot() const { return m_plot; }
  void Update(MandelbrotPlot_t const & plot);
  void Resize(unsigned w, unsigned h);
  void Location(double left, double right, double top);
  void Zoom(double const& dz);
  void Move(double dx, double dy);
private:
  MandelbrotPlot_t m_plot;
  cuMandelbrotData m_cuData;
  double   m_scale;
  uint8_t * m_buffer;
  std::size_t m_buffer_size;
};

