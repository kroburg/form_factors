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

  struct cast_result_t
  {
    float distance; // not distance but square distance
    vec3 point;
  };

  // @todo looks useless right now
  struct ray_t
  {
    vec3 origin;
    vec3 direction;
  };

  __global__ void load_scene_faces(const ray_caster::face_t* source, face_t* target, int n_faces);

  //@todo ugly copy-paste definitions. Should be dropped after algorithm optimizations.
#define TRIANGLE_INTERSECTION_UNIQUE 0
#define TRIANGLE_INTERSECTION_DISJOINT 1
#define TRIANGLE_INTERSECTION_DEGENERATE 2
#define TRIANGLE_INTERSECTION_SAME_PLAIN 3
  __device__ int triangle_intersect(ray_t ray, const vec3* triangle, vec3* point);

  // @todo Is it better to pass ray_t* instead of several vec3?
  __global__ void cast_scene_faces(const face_t* faces, int n_faces, const ray_t* rays, cast_result_t* results);
}