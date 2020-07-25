#pragma once
#include <SFML/Graphics.hpp>
#include "mandelbrot-class.hpp"
class ZoomRectangle
{
public:
  ZoomRectangle();
  ZoomRectangle(MandelbrotSet& set);
  void Initialize();
  void SetMandel(MandelbrotSet       & set) { m_set  = &set; }
  void SetOrigin(sf::Vector2f const& ori) { m_origin = ori; }
  void Update(sf::Vector2i);
  void Draw(sf::RenderWindow &);
  MandelbrotPlot_t NewPlot() const;
  bool IsActive() const { return active; }
  void SetActive(bool act = true) { active = act; }
private:
  sf::RectangleShape m_shape;
  MandelbrotSet* m_set;
  sf::Vector2f m_origin;
  bool active;
};

