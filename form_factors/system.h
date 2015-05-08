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

namespace form_factors
{
  typedef math::triangle_t face_t;

  struct mesh_t
  {
    int n_faces;
    int first_idx;
  };

  struct scene_t
  {
    int n_faces;
    face_t *faces;
    int n_meshes;
    mesh_t* meshes;
  };

  scene_t* scene_create();
  void scene_free(scene_t* s);

  struct task_t
  {
    int n_rays;
    float* form_factors;
  };

  struct system_t
  {
    const struct system_methods_t* methods;
  };

#define FORM_FACTORS_OK    0
#define FORM_FACTORS_ERROR 1

  struct system_methods_t
  {
    // @todo Add double init/shutdown check in base ray caster system.
    int(*init)(system_t* system);
    int(*shutdown)(system_t* system);

    int(*set_scene)(system_t* system, scene_t* scene);
    int(*prepare)(system_t* system);
    int(*calculate)(system_t* system, task_t* task);
  };

#define FORM_FACTORS_CPU 1

  /// @note init() system on creation.
  system_t* system_create(int type);

  /// @note shutdown() system on destruction.
  void system_free(system_t* system);

  int system_init(system_t* system);
  int system_shutdown(system_t* system);
  int system_set_scene(system_t* system, scene_t* scene);
  int system_prepare(system_t* system);
  int system_calculate(system_t* system, task_t* task);
}
