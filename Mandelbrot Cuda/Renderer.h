#pragma once
#include <deque>

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include "mandelbrot-class.hpp"
#include "zoom-rectangle.hpp"
#include "smooth-zoom.hpp"
class Renderer
{
  enum class RenderMode
  {
    Normal,
    Update,
    SmoothZoom,
    SmoothZoomReady
  };
  using uint = unsigned int;
public:
  Renderer();
  void Initialize(uint = 1440, uint = 1080);
  void Loop();
private:
  void PollEvents();
  std::unique_ptr<MandelbrotSet>
    m_set;
  sf::RenderWindow m_window;
  sf::Event m_event;
  sf::Texture front_buffer;
  sf::Texture back_buffer;
  sf::RectangleShape rect;
  sf::Texture texture;
  ZoomRectangle zoom_rect;
  std::deque<MandelbrotPlot_t> m_history;
  sf::Mouse m_mouse;
  RenderMode render_mode;
  SmoothZoom  smooth_zoom;
};

