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
  int import_obj(const char* filename, subject::scene_t** scene);
}
