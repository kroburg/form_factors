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
 * This module contains basic types to represent a scene emission calculation.
 * Scene emission generate ray-caster system rays for radiosity calculation.
 * Physical interpretation of results is out of scope of this library.
 * Module also contains base type (system_t) for emission calculation with table of virtual methods.
 */

#pragma once

#include "../math/types.h"
#include "../ray_caster/system.h"

namespace emission
{
  /// @brief Mesh type (group of polygons - as offset in scene polygons).
  struct mesh_t
  {
    int first_idx;
    int n_faces;
  };

  /// @brief Scene representation.
  /// @todo Provide single name for face/polygon/element entity. Consider that in future elements are not limited by 3D 3-point faces.
  struct scene_t
  {
    int n_faces; ///< Total number of polygons.
    math::face_t *faces; ///< Polygons array.
    int n_meshes; ///< Number of meshes.
    mesh_t* meshes; ///< Meshes array.
  };

  /// @brief Allocate memory for scene.
  scene_t* scene_create();

  /// @brief Free memory for scene.
  void scene_free(scene_t* s);

  /**
   * @brief Task representation for given scene (@see task_create).
   *
   * Each task consists of N rays (as input for ray_caster) and out-parameter rays (result of rays casting).
   */
  struct task_t
  {
    int n_rays;
    float total_weight; ///< Total scene weight.
    float* weights; ///< Two weights per face: first one in normal direction, second in opposite.
    ray_caster::task_t* rays; ///< Ray caster task (contains input and output).
  };

  /// @brief create task for given scene with n_rays rays.
  task_t* task_create(scene_t* scene, int n_rays);

  /// @brief Free memory for given task
  void task_free(task_t* task);

  /**
   * @brief Form factors calculator type.
   *
   * Represents base calculator type with virtual methods table.
   */
  struct system_t
  {
    /// @brief virtual methods table.
    const struct system_methods_t* methods;
  };

  #define EMISSION_OK    0
  #define EMISSION_ERROR 200

  /**
    @brief Callback for face weights calculation.
    @param task All memory is managed by the system.
  */
  typedef int(*calculate_weights)(scene_t* scene, task_t* task);

  /**
   *   @brief Virtual methods table for calculator functionality.
   *
   *   Every concrete calculator should implement these methods.
   *   C-interface functions (system_* functions, see below) just allocate memory and call these virtual methods.
   */
  struct system_methods_t
  { 
    /// @brief Initializes system with given ray caster after creation.
    int(*init)(system_t* system, ray_caster::system_t* ray_caster, calculate_weights weights);

    /// @brief Shutdowns calculator system prior to free memory.
    int(*shutdown)(system_t* system);

    /// @brief Sets loaded scene (polygons in meshes) for calculator and associated ray caster.
    int(*set_scene)(system_t* system, scene_t* scene);

    /// @brief Calculates weights, generate emission rays and casts rays for given system.
    int(*calculate)(system_t* system, task_t* task);
  };

  #define EMISSION_CPU 1

  /**
   * @brief Factory method for calculator creation.
   *
   * Creates emission calculator system with given calculator type and raycaster.
   * @note Only CPU calculator type (type = 1) is supported, @see ../cpuEmission/.
   * @note init() system on creation.
   */
  system_t* system_create(int type, ray_caster::system_t* ray_caster, calculate_weights weights);

  /// Here go C-interface wrappers to call system_t's virtual methods.

  /// @note shutdown() system on destruction.
  void system_free(system_t* system);
  int system_init(system_t* system, ray_caster::system_t* ray_caster, calculate_weights weights);
  int system_shutdown(system_t* system);
  int system_set_scene(system_t* system, scene_t* scene);
  int system_calculate(system_t* system, task_t* task);
}
