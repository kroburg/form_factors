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
 * This module contains basic types to represent a thermal equation.
 */

#pragma once

#include "../math/types.h"
#include "../thermal_solution/system.h"

namespace thermal_equation
{
  struct task_t
  {
    const float* temperatures; ///< Objects temperature array (K). @note Not owned by equation.
    float* emission; ///< Energy emission array per object (mesh) (W). Equation must add to task values, not set to zero.
    float* absorption; ///< Energy absorption array per object (mesh) (W). Equation must add to task values, not set to zero.
  };

  task_t* task_create(subject::scene_t* scene);
  void task_free(task_t* task);

  /**
   * @brief Thermal equation type represents a right side of thermal solution problem.
   */
  struct system_t
  {
    /// @brief virtual methods table.
    const struct system_methods_t* methods;
  };

#define THERMAL_EQUATION_OK              0
#define THERMAL_EQUATION_ERROR           400

  /**
   *   @brief Virtual methods table for calculator functionality.
   *
   *   Every concrete calculator should implement these methods.
   *   C-interface functions (system_* functions, see below) just allocate memory and call these virtual methods.
   */
  struct system_methods_t
  {
    /// @brief Initializes equation.
    int(*init)(system_t* system, void* params);

    /// @brief Shutdowns calculator system prior to free memory.
    int(*shutdown)(system_t* system);

    /// @brief Sets loaded scene (polygons in meshes) for calculator.
    /// @note System does not own scene object.
    int(*set_scene)(system_t* system, subject::scene_t* scene);

    /// @brief Calculates energy flow for given system.
    int(*calculate)(system_t* system, task_t* task);
  };

  /// @brief Stefan-Boltzman radiation equation spread by pre-calculated form factors.
  #define THERMAL_EQUATION_SB_FF_CPU 1
  #define THERMAL_EQUATION_RADIANCE_CPU 2
  #define THERMAL_EQUATION_CONDUCTIVE_CPU 3
  #define THERMAL_EQUATION_HEAT_SOURCE_CPU 4
  #define THERMAL_EQUATION_DISNTANT_SOURCE_CPU 5

  /**
   * @brief Factory method for equation creation.
   * @note init() system on creation.
   */
  system_t* system_create(int type, void* params);

  /// Here go C-interface wrappers to call system_t's virtual methods.

  /// @note shutdown() system on destruction.
  void system_free(system_t* system);
  int system_init(system_t* system, void* params);
  int system_shutdown(system_t* system);
  int system_set_scene(system_t* system, subject::scene_t* scene);
  int system_calculate(system_t* system, task_t* task);
}
