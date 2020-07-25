#include "mandelbrot-class.hpp"
#include <chrono>
#include <iostream>
MandelbrotSet::MandelbrotSet()
  : m_plot(1440, 1080, MAX_DIMENSIONS[0], MAX_DIMENSIONS[1], MAX_DIMENSIONS[2], MAX_DIMENSIONS[3])
  , m_scale(1.0)
  , m_buffer_size(1440 * 1080 * W_PIXEL)
  , m_buffer(new uint8_t[1440 * 1080 * W_PIXEL]())
{
}
MandelbrotSet::MandelbrotSet(std::size_t const & x, std::size_t const & y)
  : m_plot(x, y, MAX_DIMENSIONS[0], MAX_DIMENSIONS[1], MAX_DIMENSIONS[2], MAX_DIMENSIONS[3])
  , m_scale(1.0)
  , m_buffer_size(x * y * W_PIXEL)
  , m_buffer(new uint8_t[x * y * W_PIXEL]())
  , m_cuData()
{
  InitializeCudaData(&m_cuData, m_buffer_size);
}

uint8_t const* MandelbrotSet::Render()
{

  auto start = std::chrono::steady_clock::now();
  MandelbrotCudaCall(m_buffer, m_plot, m_cuData);
  auto end = std::chrono::steady_clock::now();
  std::cout << "Time to generate " << " = " << std::chrono::duration <double, std::milli>(end - start).count() << " [ms]" << std::endl;
  return m_buffer;
}

void MandelbrotSet::Render(MandelbrotPlot_t const& plot, uint8_t* ext_buffer)
{
  MandelbrotCudaCall(ext_buffer, plot, m_cuData);

}

void MandelbrotSet::Update(MandelbrotPlot_t const& plot)
{
  m_plot = plot;
}

void MandelbrotSet::Resize(unsigned w, unsigned h)
{
  const double aspect_ratio = (static_cast<double>(h) / static_cast<double>(w));
  
}

void MandelbrotSet::Zoom(double const& dz)
{
  //Old Code
  /*
  const std::array<double, 2> PLOT_CENTER = m_plot.GetCenter();
  const double ASPECT_RATIO = m_plot.width / m_plot.height;
  double x_width;
  double y_height;
  m_scale  = std::max(1.0, m_scale + dz);
  x_width  = (MAX_DIMENSIONS[1] - MAX_DIMENSIONS[0]) / m_scale / 2.0;
  y_height = ((MAX_DIMENSIONS[2] - MAX_DIMENSIONS[3]) / ASPECT_RATIO) / m_scale / 2.0;
  m_plot.x_min = PLOT_CENTER[0] - x_width;
  m_plot.x_max = PLOT_CENTER[0] + x_width;
  m_plot.y_bot= PLOT_CENTER[1] - y_height;
  m_plot.y_top= PLOT_CENTER[1] + y_height;
  std::cout << "Scale: " << m_scale << "\n";
  */
  const auto PLOT_CENTER    = m_plot.GetCenter();
  const auto PLOT_DIM       = m_plot.GetDimensions();
  const double ASPECT_RATIO = m_plot.width / m_plot.height;
  double x_width;
  double y_height;
  x_width  = (PLOT_DIM[0] * dz) / 2.0;
  y_height = (PLOT_DIM[1] * dz) / 2.0;
  m_plot.x_min = PLOT_CENTER[0] - x_width;
  m_plot.x_max = PLOT_CENTER[0] + x_width;
  m_plot.y_bot = PLOT_CENTER[1] + y_height;
  m_plot.y_top = PLOT_CENTER[1] - y_height;
}

void MandelbrotSet::Move(double xdt, double ydt)
{
  const double dx = (m_plot.x_max - m_plot.x_min) * xdt;
  const double dy = (m_plot.y_bot - m_plot.y_top) * ydt;
  std::cout << "Plot:  " << m_plot << "\n" << 
    "Delta: " << "{" << xdt << ", " << ydt << "}, {" << dx << ", " << dy << "}\n";
  m_plot.x_max += dx;
  m_plot.x_min += dx;
  m_plot.y_bot += dy;
  m_plot.y_top += dy;
  std::cout << "Plot:  " << m_plot << "\n";
}
