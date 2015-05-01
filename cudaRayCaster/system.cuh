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

  struct scene_t
  {
    int n_faces;
    face_t *faces;
  };

  __global__ void load_scene_faces(const ray_caster::face_t* source, face_t* target, int n_faces);
}