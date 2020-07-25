#include "Renderer.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include "cudaImmediary.h"
#include "mUtilities.h"
Renderer::Renderer()
  : m_window()
  , m_event()
  , rect()
  , m_set()
  , render_mode(RenderMode::Normal)
  , zoom_rect()
  , m_mouse()
  , m_history()
  , smooth_zoom()
{
}

void Renderer::Initialize(uint width, uint height)
{
  std::array<double, 10> arr2;
  arr2.fill(10);
  std::array<std::array<double, 10>, 10> arr;
  arr.fill(arr2);
  std::cout << arr << '\n';
  m_window.create(sf::VideoMode(width, height), "Mandelbrot");
  texture.create(width, height);
  m_set = std::make_unique<MandelbrotSet>(width, height);
  m_set->Render();
  texture.update(m_set->GetBuffer(), width, height, 0, 0);
  rect.setSize(sf::Vector2f(width, height));
  rect.setTexture(&texture);
  rect.setPosition(0, 0);
  zoom_rect.SetMandel(*m_set);
  m_window.setFramerateLimit(30);
  m_window.setKeyRepeatEnabled(true);
  m_history.push_front(m_set->GetPlot());
  //rect.setTexture(&test_texture);
}

void Renderer::Loop()
{
  int half_frame = 0;
  while (m_window.isOpen())
  {
    PollEvents();
    switch (render_mode)
    {
    case RenderMode::Update:
        m_set->Render();
        texture.update(m_set->GetBuffer());
        render_mode = RenderMode::Normal;
        break;
    case RenderMode::SmoothZoom:
      if (half_frame & 1)
      {
        ++half_frame;
        break;
      }
      if (smooth_zoom.Iterator() == smooth_zoom.GetBuffers().cend())
      {
        smooth_zoom.ClearBuffers();
        render_mode = RenderMode::Normal;
        break;
      }
      texture.update(*(smooth_zoom.Iterator()++));
      break;

    }
    m_window.clear();
    m_window.draw(rect);
    if (zoom_rect.IsActive()) {
      zoom_rect.Update(m_mouse.getPosition(m_window));
      zoom_rect.Draw(m_window);
    }

    m_window.display();
  }
}

void Renderer::PollEvents()
{
  while (m_window.pollEvent(m_event))
  {
    if (m_event.type == sf::Event::Closed)
    {
      m_window.close();
    }
    else if (m_event.type == sf::Event::Resized)
    {

    }
    else if (m_event.type == sf::Event::KeyPressed)
    {

      if (render_mode == RenderMode::SmoothZoomReady && m_event.key.code == sf::Keyboard::Enter)
      {
        render_mode = RenderMode::SmoothZoom;
      }
      else if (render_mode == RenderMode::Normal)
      {
        switch (m_event.key.code) {
        case sf::Keyboard::Enter:
          if (render_mode == RenderMode::SmoothZoomReady)
            render_mode = RenderMode::SmoothZoom;
          break;
        case sf::Keyboard::Z:
          if (m_event.key.shift == true)
          {
            m_set->Zoom((1 / .75));
            m_history.push_front(m_set->GetPlot());
          }
          else if (m_event.key.control == true)
          {
            if (m_history.empty() || m_history.size() == 1)
              break;
            m_set->Update(m_history.front());
            m_history.pop_front();
          }
          else {
            m_set->Zoom(.75);
            m_history.push_front(m_set->GetPlot());
          }
          render_mode = RenderMode::Update;
          break;
        case sf::Keyboard::Left:
          m_set->Move(-.25, 0.0);
          render_mode = RenderMode::Update;
          m_history.push_front(m_set->GetPlot());
          break;
        case sf::Keyboard::Right:
          m_set->Move(.25, 0.0);
          render_mode = RenderMode::Update;
          m_history.push_front(m_set->GetPlot());
          break;
        case sf::Keyboard::Up:
          m_set->Move(0.0, .25);
          render_mode = RenderMode::Update;
          m_history.push_front(m_set->GetPlot());
          break;
        case sf::Keyboard::Down:
          m_set->Move(.0, -.25);
          render_mode = RenderMode::Update;
          m_history.push_front(m_set->GetPlot());
          break;
        }
      }
    }
    else if (m_event.type == sf::Event::MouseButtonPressed && render_mode == RenderMode::Normal)
    {
      if (m_event.mouseButton.button == sf::Mouse::Left)
      {
        zoom_rect.SetOrigin(sf::Vector2f(static_cast<float>(m_mouse.getPosition(m_window).x), static_cast<float>(m_mouse.getPosition(m_window).y)));
        zoom_rect.SetActive();
      }
    }
    else if (m_event.type == sf::Event::MouseButtonReleased && render_mode == RenderMode::Normal)
    {
      if (m_event.mouseButton.button == sf::Mouse::Left)
      {
        if (m_event.key.control == true)
        {
          render_mode = RenderMode::SmoothZoomReady;
          smooth_zoom.Initialize(m_set->GetPlot(), zoom_rect.NewPlot(), 64);
          smooth_zoom.Render(*m_set);
          zoom_rect.SetActive(false);
          std::cout << "Ready\n";
        }
        else {
          m_set->Update(zoom_rect.NewPlot());
          zoom_rect.SetActive(false);
          render_mode = RenderMode::Update;
        }
      }
    }
  }
}
