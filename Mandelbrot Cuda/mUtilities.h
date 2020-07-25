#pragma once
#include <array>
#include <iostream>
template<typename T, std::size_t S>
std::ostream& operator<< (std::ostream& os, std::array<T, S> const& arr)
{
  os << "{";
  for (std::size_t i = 0; i < S - 1; ++i)
  {
    os << arr[i] << ", ";
  }
  os << arr.back() << "}";
  return os;
}