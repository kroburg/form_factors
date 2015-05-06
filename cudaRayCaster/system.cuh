# Copyright 2015 Stepan Tezyunichev (stepan.tezyunichev@gmail.com).
# This file is part of form_factors.
#
# form_factors is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# form_factors is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with form_factors.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include "../ray_caster/system.h"
#include <helper_math.h>

namespace cuda_ray_caster
{  
  typedef float3 vec3;

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

  __global__ void load_scene_faces(const ray_caster::face_t* source, face_t* target, int n_faces);
  __global__ void load_rays(const math::ray_t* source, ray_t* target, int n_rays);

  __device__ bool face_bbox_intersect(ray_t ray, const face_t* face);

  //@todo ugly copy-paste definitions. Should be dropped after algorithm optimizations.
#define TRIANGLE_INTERSECTION_UNIQUE 0
#define TRIANGLE_INTERSECTION_DISJOINT 1
#define TRIANGLE_INTERSECTION_DEGENERATE 2
#define TRIANGLE_INTERSECTION_SAME_PLAIN 3

  __device__ int triangle_intersect(ray_t ray, const vec3* triangle, vec3* point);

  __global__ void cast_scene_faces_with_reduction(const face_t* faces, int n_faces, const ray_t* rays, int* indecies, vec3* points);
}
