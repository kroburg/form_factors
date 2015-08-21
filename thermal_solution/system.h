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
 * This module contains basic types to thermal solution calculation.
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
   * @param temperatures current meshes temperatures.
   */
  struct task_t
  {
    int n_step; ///< Current integration step.
    float time_delta;
    float* temperatures;  ///< Result temperatures.
  };

  task_t* task_create(scene_t* scene);
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
    /// @todo Provide (back again) prepare() call for initial (time consuming calculations). Consider form-factors calculation for Stefan-Boltzman form-factors based thermal equation.
    int(*set_scene)(system_t* system, scene_t* scene, float* temperatures);

    /// @brief Calculates thermal solution.
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
  int system_set_scene(system_t* system, scene_t* scene, float* temperatures);
  int system_calculate(system_t* system, task_t* task);
}
