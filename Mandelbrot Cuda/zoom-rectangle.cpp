#include "zoom-rectangle.hpp"
#include <iostream>
#include "mUtilities.h"
ZoomRectangle::ZoomRectangle()
  : active(false)
  , m_shape({ 100.f, 100.f })
  , m_set(nullptr)
{
  Initialize();
}

ZoomRectangle::ZoomRectangle(MandelbrotSet& set)
  : active(false)
  , m_shape()
  , m_set(&set)
{
  Initialize();
}

void ZoomRectangle::Initialize()
{
  m_shape.setFillColor(sf::Color::Transparent);
  m_shape.setOutlineColor(sf::Color(0xff, 0xff, 0xff, 0x7F));
  m_shape.setOutlineThickness(2.0f);
}

void ZoomRectangle::Draw(sf::RenderWindow& window)
{
  if (active) 
    window.draw(m_shape);
}

MandelbrotPlot_t ZoomRectangle::NewPlot() const
{
  const auto size = m_shape.getSize();
  const auto curr_plot = m_set->GetPlot();
  const auto plot_dim    = curr_plot.GetDimensions();
  MandelbrotPlot_t new_plot = curr_plot;
  std::array<double, 2> new_center = 
  { 
    plot_dim[0] * (m_origin.x / curr_plot.width) + curr_plot.x_min,
    plot_dim[1] * (m_origin.y / curr_plot.height) + curr_plot.y_top 
  };
  std::cout << "curr:   " << curr_plot << '\n';
  std::cout << "Rect:   " << m_origin.y << ", " << curr_plot.height << '\n';
  std::cout << "Center: " << new_center << '\n';
  double delta = (size.x / curr_plot.width) * plot_dim[0] / 2.0;
  new_plot.x_max = new_center[0] + delta;
  new_plot.x_min = new_center[0] - delta;
  new_plot.y_bot = new_center[1] + delta * m_set->GetAspect();
  new_plot.y_top = new_center[1] - delta * m_set->GetAspect();
  return new_plot;
}

void ZoomRectangle::Update(const sf::Vector2i mouse_pos)
{
  const float aspect_ratio = m_set->GetAspect();
  const float delta(std::abs(static_cast<float>(mouse_pos.x - m_origin.x)) * 2.0);
  const sf::Vector2f size(delta, delta * aspect_ratio);
  m_shape.setSize(size);
  m_shape.setPosition(m_origin - (size / 2.0f));
}
