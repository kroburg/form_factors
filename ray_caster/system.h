#pragma once

#include <stdlib.h>

namespace ray_caster
{
  struct vec3
  {
    float x;
    float y;
    float z;
  };

  struct face_t
  {
    vec3 points[3];
  };

  inline face_t make_face(vec3 a, vec3 b, vec3 c)
  {
    return { a, b, c };
  }

  struct ray_t
  {
    vec3 origin;
    vec3 direction;
  };

  struct task_t
  {
    int n_rays;
    ray_t* rays;
    face_t* hit_face; // 0 if no hit
    vec3 hit_point;
  };

  struct scene_t
  {
    int n_faces;
    face_t *faces;
  };

  scene_t* scene_create();
  void scene_free(scene_t* s);

  struct system_t
  {
    const struct system_methods_t* methods;
  };

#define RAY_CASTER_OK    0
#define RAY_CASTER_ERROR 1
#define RAY_CASTER_OUT_OF_RANGE 2

  struct system_methods_t
  {
    // @todo Add double init/shutdown check in base ray caster system.
    int(*init)(system_t* system);
    int(*shutdown)(system_t* system);

    // Accept scene ownership.
    int (*set_scene)(system_t* system, scene_t* scene);

    // Prepare system for ray casting.
    int (*prepare)(system_t* system);
  };  

#define RAY_CASTER_SYSTEM_CPU 1
#define RAY_CASTER_SYSTEM_CUDA 2

  /// @note init() system on creation.
  system_t* system_create(int type);
  system_t* system_create_default();
  
  /// @note shutdown() system on destruction.
  void system_free(system_t* system);

  int system_init(system_t* system);
  int system_shutdown(system_t* system);
  int system_set_scene(system_t* system, scene_t* scene);
  int system_prepare(system_t* system);
}