#include <cstdint>
#include "Renderer.h"
int main(int argc, char* argv[])
{
  Renderer renderer;
  renderer.Initialize(2700, 1800);
  renderer.Loop();
  return 0;
}