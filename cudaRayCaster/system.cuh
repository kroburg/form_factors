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
 * This module contains GPU-oriented methods to cast rays on Cuda devices.
 * For ray_caster::system_t implementation see cuda_system.h and system.cu files.
 */

#pragma once

#include "../ray_caster/system.h"
#include <helper_math.h>

namespace cuda_ray_caster
{
  /// @brief Redefinition for device type.
  typedef float3 vec3;

  /// @brief Adding bounding box to face type.
  struct face_t
  {
    vec3 points[3];
    vec3 bbox[2];
  };

  // @todo looks useless right now
  struct ray_t
  {
    vec3 origin;
    vec3 direction;
    vec3 inv_dir;
  };

  /// @brief Loads scene to GPU and converts base face_t to cuda_ray_caster::face_t (with bounding box).
  __global__ void load_scene_faces(const ray_caster::face_t* source, face_t* target, int n_faces);

  /// @brief Loads task with n_rays to GPU.
  __global__ void load_rays(const math::ray_t* source, ray_t* target, int n_rays);

  /// @brief Checks face's bounding box intersection with ray (optimization prior to true intersection).
  __device__ bool face_bbox_intersect(ray_t ray, const face_t* face);

  //@todo ugly copy-paste definitions. Should be dropped after algorithm optimizations.
  #define TRIANGLE_INTERSECTION_UNIQUE 0
  #define TRIANGLE_INTERSECTION_DISJOINT 1
  #define TRIANGLE_INTERSECTION_DEGENERATE 2
  #define TRIANGLE_INTERSECTION_SAME_PLAIN 3

  /**
   * @brief Check ray and triangle intersection (device-oriented duplication of math's library function).
   * @param[in] ray Ray to intersect.
   * @param[in] triangle Face to intersect.
   * @param[out] Vector to ray and triangle intersection (0 if no intersection).
   */
  __device__ int triangle_intersect(ray_t ray, const vec3* triangle, vec3* point);

  /**
   * @brief Cast rays on scene with n_faces.
   * @param[in] faces[in] Array of faces.
   * @param[in] n_faces[in] Number of faces in faces array.
   * @param[in] rays Array of rays (kernel determines needed ray by blockIdx.x)
   * @param[out] indices Array of intersection results with each face (-1 if no intersection or face index in faces array if intersected).
   * @param[out] points Array of intersection points for each face in faces array.
   */
  __global__ void cast_scene_faces_with_reduction(const face_t* faces, int n_faces, const ray_t* rays, int* indices, vec3* points);
}
