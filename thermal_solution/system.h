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
 * This module contains basic types to represent a scene for form factors calculation.
 * Module also contains base type (system_t) for form factor calculation with table of virtual methods.
 */

#pragma once

#include "../math/types.h"

namespace thermal_equation
{
  struct system_t;
}

namespace thermal_solution
{
  /// @brief Face (polygon) type
  typedef math::triangle_t face_t;

  struct params_t
  {
    int n_equations;
    struct thermal_equation::system_t* equations;
  };

  struct material_t
  {
    float c; // dQ/dT per square meter
  };

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
    face_t *faces; ///< Polygons array.    
    int n_meshes; ///< Number of meshes.
    mesh_t* meshes; ///< Meshes array.    
    int n_materials; ///< Total number of materials.
    material_t* materials; ///< Materials array.
  };

  /// @brief Allocate memory for scene.
  scene_t* scene_create();

  /// @brief Free memory for scene.
  void scene_free(scene_t* s);

  /**
   * @brief Task representation for given scene (@see task_create).
   * @detail Calculate thermal solution using Adams integration method.
   * @param temperatures current meshes temperatures.
   */
  struct task_t
  {
    float* temperatures; ///< Initial and result meshes temperatues.
    float time_delta;
    int n_step; ///< Current integration step. Must be set to zero for first iteration.
    float* previous_temperatures; ///< Previous four Adams method values.
  };

  /// @brief create task for given scene. Allocate memory for temperatues array.
  task_t* task_create(scene_t* scene);

  /// @brief Free memory for given task.
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

#define THERMAL_SOLUTION_OK              0
#define THERMAL_SOLUTION_ERROR           300

  /**
   *   @brief Virtual methods table for calculator functionality.
   *
   *   Every concrete calculator should implement these methods.
   *   C-interface functions (system_* functions, see below) just allocate memory and call these virtual methods.
   */
  struct system_methods_t
  {
    // @todo Add double init/shutdown check in base system.

    /// @brief Initializes system with given equations.
    int(*init)(system_t* system, params_t* params);

    /// @brief Shutdowns calculator system prior to free memory.
    int(*shutdown)(system_t* system);

    /// @brief Sets scene (polygons in meshes) for calculator and all equations.
    int(*set_scene)(system_t* system, scene_t* scene);

    /**
     *  @brief Calculates thermal solution.
     *
     */
    int(*calculate)(system_t* system, task_t* task);
  };

  #define THERMAL_SOLUTION_CPU_ADAMS 1

  /**
   * @brief Factory method for calculator creation.
   *
  */
  system_t* system_create(int type);

  /// Here go C-interface wrappers to call system_t's virtual methods.

  /// @note shutdown() system on destruction.
  void system_free(system_t* system);
  int system_init(system_t* system, params_t* params);
  int system_shutdown(system_t* system);
  int system_set_scene(system_t* system, scene_t* scene);
  int system_calculate(system_t* system, task_t* task);
}
