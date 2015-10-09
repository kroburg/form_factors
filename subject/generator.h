// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
// This file is part of form_factors.
//
// form_factors is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// form_factors is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with form_factors.  If not, see <http://www.gnu.org/licenses/>.

/**
* This module contains generator uniformly distributed points generators.
*/

#include "../math/types.h"

namespace subject
{
  /// @todo Move out from spherical_generator.h
  struct generator_t
  {
    /// @brief virtual methods table.
    const struct generator_methods_t* methods;
  };

  /// @brief Virtual methods table for generator functionality.
  struct generator_methods_t
  {
    /// @brief Initializes generator.
    int(*init)(generator_t* generator);

    /// @brief Shutdown and free generator memory.
    void(*shutdown)(generator_t* generator);

    /// @brief Generate surface point.
    int(*surface_point)(generator_t* generator, int count, math::vec3* result);

    /// @brief Generate volume point.
    int(*volume_point)(generator_t* generator, int count, math::vec3* result);
  };

  generator_t* generator_create_spherical();
  generator_t* generator_create_cube();

  int generator_init(generator_t* generator);
  void generator_free(generator_t* generator);
  int generator_surface_point(generator_t* generator, int count, math::vec3* result);
  int generator_volume_point(generator_t* generator, int count, math::vec3* result);
}