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
* This module contains thermal equation for heat sources and sinks.
*/

#pragma once

#include "../thermal_equation/system.h"
#include "../emission/system.h"

namespace heat_source_equation
{
  struct heat_source_t
  {
    int mesh_idx;

    /// @note Power value may be negative. Then heat source become a heat sink.
    float power;
  };

  struct params_t
  {
    int n_sources;
    heat_source_t* sources;
  };

  thermal_equation::system_t* system_create();
}
