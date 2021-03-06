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
 * This module contains basic types to represent a scene for ray caster calculation.
 * Module also contains base type (system_t) for ray caster with table of virtual methods.
 */

#pragma once

#include "../math/types.h"

namespace ray_caster
{
  /// @todo: Why? Sources use math prefix.
  typedef math::triangle_t face_t;
  typedef math::ray_t ray_t;

  /**
   * @brief Ray caster task representation.
   * @detail Each task consists of n_tasks rays (input), indices of hit scene faces and hit points (output variables).
   * If ray has no intersection with scene face it index will be -1.
   */
  struct task_t
  {
    int n_tasks;
    math::ray_t* ray;
    face_t** hit_face;
    math::vec3* hit_point;
  };

  /// @brief Scene representation.
  struct scene_t
  {
    int n_faces;
    face_t *faces;
  };

  /// @brief Allocate memory for scene.
  scene_t* scene_create();

  /// @brief Free memory for scene.
  void scene_free(scene_t* s);

  /// @brief Allocates memory for task with n_rays rays.
  task_t* task_create(int n_rays);

  /// @brief Frees memory for task.
  void task_free(task_t* task);

  /**
   * @brief Ray caster type.
   *
   * Represents base type for ray caster implementations with virtual methods table.
   */
  struct system_t
  {
    /// @brief virtual methods table.
    const struct system_methods_t* methods;
  };

  #define RAY_CASTER_OK    0
  #define RAY_CASTER_ERROR 1
  #define RAY_CASTER_NOT_SUPPORTED 2

  /**
   *   @brief Virtual methods table for ray caster functionality.
   *
   *   Every concrete calculator should implement these methods.
   *   C-interface functions (system_* functions, see below) just allocate memory and call these virtual methods.
   */
  struct system_methods_t
  {
    // @todo Add double init/shutdown check in base ray caster system.

    /// @brief Initializes system after creation.
    int(*init)(system_t* system);

    /// @brief Shutdowns system prior to free memory.
    int(*shutdown)(system_t* system);

    /// @brief Sets loaded scene (polygons in meshes) for ray caster.
    int(*set_scene)(system_t* system, scene_t* scene);

    /// @brief Checks system consistency and prepare scene for ray casting.
    int(*prepare)(system_t* system);

    /// @brief Casts rays of given task task for prepared scene.
    /// @note Task's rays, rays hit faces indices and points are prepared by callee.
    int(*cast)(system_t* system, task_t* task);
  };

  #define RAY_CASTER_NAIVE_CPU 1
  #define RAY_CASTER_NAIVE_CUDA 2
  #define RAY_CASTER_ZGRID_CPU 3
  #define RAY_CASTER_ZGRID_CUDA 4

  /**
   * @brief Factory method for ray caster creation.
   * @param type[in] @see RAY_CASTER_NAIVE_CPU for CPU caster and @see RAY_CASTER_NAIVE_CUDA for GPU.
   * @note init() system on creation.
   */
  system_t* system_create(int type);

  /// @brief Creates default ray caster system (CPU).
  system_t* system_create_default();
  
  /// @brief Frees ray caster resources.
  /// @note shutdown() system on destruction.
  void system_free(system_t* system);

  /// Here go C-interface wrappers to call system_t's virtual methods.

  int system_init(system_t* system);
  int system_shutdown(system_t* system);
  int system_set_scene(system_t* system, scene_t* scene);
  int system_prepare(system_t* system);
  int system_cast(system_t* system, task_t* task);
}
