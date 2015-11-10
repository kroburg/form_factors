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


#pragma once

#include "../math/types.h"
#include "system.h"

namespace subject
{
  const math::face_t* box(); // 12 faces packed by two faces per box face.
  math::face_t* plane_grid_faces(float width, float height, int cells_x, int cells_y); // allocate and intialize plane grid
  mesh_t* plane_grid_meshes(int cells_x, int cells_y);
  math::face_t* tplane_grid_faces(float width, float height, int cells_x, int cells_y);
  mesh_t* tplane_grid_meshes(int cells_x, int cells_y);
  
  material_t material_Al(float thickness);
}
