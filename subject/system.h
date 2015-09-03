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
* This module contains basic types for scene definition and manipulation.
* @todo Provide correct file name instead of system.h
*/

#pragma once

#include "../math/types.h"

namespace subject
{
  /// @brief Face (polygon) type
  typedef math::triangle_t face_t;
  
  struct shell_properties_t
  {
    float density; ///< kg/m^3
    float heat_capacity; ///< dQ/dT J/(kg*K)
    float thermal_conductivity; ///< W/(m*K)
    float thickness; ///< m
  };

  shell_properties_t default_shell_properties();

  struct optical_properties_t
  {
    float specular_reflectance;
    float diffuse_reflectance;
    float absorbance; ///< ����������� ���������� (rus.)
    float transmittance; ///< The ratio of the light energy falling on a body to that transmitted through it. ����������� ����������� (rus.)
    float emissivity; ///< ������� ������� (rus.)
  };

  optical_properties_t black_material();

  struct material_t
  {
    shell_properties_t shell;
    optical_properties_t front;
    optical_properties_t rear;
  };

  /// @brief Black body with major parameters equal to 1.f
  material_t black_body();

  /// @brief Mesh type (group of polygons - as offset in whole scene polygons).
  struct mesh_t
  {
    int first_idx;
    int n_faces;
    int material_idx;
  };

  /// @brief Scene representation.
  struct scene_t
  {
    int n_faces; ///< Total number of polygons.
    face_t* faces; ///< Polygons array.    
    int n_meshes; ///< Number of meshes.
    mesh_t* meshes; ///< Meshes array.    
    int n_materials; ///< Total number of materials.
    material_t* materials; ///< Materials array.
  };

  /// @brief Allocate memory for scene.
  scene_t* scene_create();

  /// @brief Free memory for scene.
  void scene_free(scene_t* s);
}
