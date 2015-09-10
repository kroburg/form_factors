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
    float absorbance; ///< Коэффициент поглощения (rus.)
    float transmittance; ///< The ratio of the light energy falling on a body to that transmitted through it. Коэффициент пропускания (rus.)
    float emissivity; ///< Степень черноты (rus.)
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

  float mesh_area(const scene_t* scene, const mesh_t& mesh);
  float mesh_area(const scene_t* scene, int mesh_idx);

  /// @brief Calculate areas for all meshes.
  /// @return Total area.
  /// @detail Reallocate target areas array memory.
  float build_meshes_areas(const scene_t* scene, float** areas); 

  /// @brief Calculate areas for all faces.
  /// @detail Reallocate target areas array memory.
  void build_faces_areas(const scene_t* scene, float** areas);

  /// @brief Build face to mesh index.
  /// @detail Reallocate target index array memory.
  void build_face_to_mesh_index(int n_faces, int n_meshes, const mesh_t* meshes, int** index);

  const material_t& mesh_material(const scene_t* scene, int mesh_idx);

  /**
    @brief Face graph walker callback.
    @param current_idx Zero-based face index.
    @param leaf_idx Zero-based face adjacent to current face. -1 if there are no adjacent faces to current face.
    @param have_more Signal if there are more adjacent faces.
    @param param Data passed to initial walk() function.
    @return 0 to continue, >0 to stop iteration, <0 for error.
  */
  typedef int(*face_graph_walker)(int current_idx, int leaf_idx, bool have_more, void* param);

  /**
    @brief Walk face graph.
    @note Algorithm complexity is ~N^2, there N is faces count.
  */
  int face_walk_graph_n2c(const face_t* faces, int n_faces, face_graph_walker walker, void* param);

  /**
    @brief Try to unify face graph normals direction.
    @note It may be impossible to to unify normals for not closed mesh (consider Mobius strip).
    @return Count on flipped faces.
  */
  int face_unify_normals(face_t* faces, int n_faces);

  /**
    @brief Try to unify mesh normals direction.
    @note It may be impossible to to unify normals for not closed mesh (consider Mobius strip).
    @return Count on flipped faces.
  */
  int mesh_unify_normals(scene_t* scene, int mesh_idx);
}
