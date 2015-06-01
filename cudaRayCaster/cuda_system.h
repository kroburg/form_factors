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
 * This module contains GPU implementation of ray_caster::system_t.
 */

#pragma once

#include "../ray_caster/system.h"

namespace cuda_ray_caster
{
  /** 
 * @brief Creates base system for GPU ray caster.
 * @detail Use Axes Aligned Bounding Box optimization.
 * @todo AABB calculated during set_scene() call but it should be done during prepare() call.
 * @todo No CUDA related calls should be performed until prepare()/cast() calls.
 * */
  ray_caster::system_t* system_create();
}
