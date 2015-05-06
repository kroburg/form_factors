// Copyright 2015 Stepan Tezyunichev (stepan.tezyunichev@gmail.com).
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

namespace ray_caster
{
  typedef math::triangle_t face_t;  

  inline face_t make_face(math::vec3 a, math::vec3 b, math::vec3 c)
  {
    return { a, b, c };
  }

  struct task_t
  {
    int n_tasks;
    math::ray_t* ray;
    face_t** hit_face;
    math::vec3* hit_point;
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
#define RAY_CASTER_NOT_SUPPORTED 2

  struct system_methods_t
  {
    // @todo Add double init/shutdown check in base ray caster system.
    int(*init)(system_t* system);
    int(*shutdown)(system_t* system);

    int(*set_scene)(system_t* system, scene_t* scene);
    int(*prepare)(system_t* system);
    int(*cast)(system_t* system, task_t* task);
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
  int system_cast(system_t* system, task_t* task);
}
