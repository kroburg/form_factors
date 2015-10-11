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
 * This module contains functionality of loading obj-files (Wavefront).
 */

#pragma once

#include "../subject/system.h"
#include "../thermal_solution/system.h"
#include "../thermal_equation/heat_source_cpu.h"
#include "../thermal_equation/parallel_rays_cpu.h"
#include <cstdio>

namespace obj_import
{
#define OBJ_IMPORT_OK 0
#define OBJ_IMPORT_FILE_ERROR 21
#define OBJ_IMPORT_FORMAT_ERROR 22
#define OBJ_IMPORT_MATERIAL_NOT_DEFINED 23
#define OBJ_IMPORT_MATERIAL_NOT_ENOUGH_PARAMETERS 24
#define OBJ_IMPORT_MATERIAL_INVALID_PARAMETER_VALUE 25

  /**
   * @brief Loads scene from obj-file.
   * @return @see OBJ_IMPORT_OK if ok, @see OBJ_IMPORT_FILE_ERROR if file can not be opened,
   * or @see OBJ_IMPORT_FORMAT_ERROR if there is format errors.
   */
  int scene(const char* filename, subject::scene_t** scene);

  /**
    @brief Load thermal solution task values from file of obj-like format.
    @param heat_source parameter values must have valid (initialized) values before call.
    @distant_source  parameter values must have valid (initialized) values before call.
    @detail obj-like format is:
      newfrm <frame name> Start of new (results) frame or task.
      step <time step> Integration step in seconds (s)
      tmprt <mesh temeprature> List of mesh temperatures (K) from first to last.
      [optional] dstsrc Distant (parallel rays) radiocity heat source (W/m^2). dstsrc <power> <x direction> <y direction> <z direction>. Direction of (-1, 0, 0) is like source is left to the scene.
      [optional] htsrc Heat source or sink (W). htsrc <mesh zero-based index> <power>.
  */
  int task(FILE* in, int n_meshes, thermal_solution::task_t* t, heat_source_equation::params_t* heat_source, parallel_rays_cpu::params_t* distant_source);
}
