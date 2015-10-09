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
* This module contains emission generator for parallel rays.
*/

#pragma once

#include "../emission/system.h"

namespace parallel_rays_emission_cpu
{
  /// @todo Provide a way to change origin and direction in time.
  struct task_t : emission::task_t
  { 
    math::vec3 direction; // in relative coordinates
    float distance;
    float height;
    float width;
  };

  /**
  *  @brief Factory method to create parallel rays emission.
  *  @detail Emit randomly (uniformly) distributed parallel rays from specified plane in direction of plane normal.
  */
  emission::system_t* system_create();
}
