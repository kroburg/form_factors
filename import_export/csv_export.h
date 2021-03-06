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
 * This module contains functionality of exporting calculated form factors to csv files.
 */

#pragma once

#include "../form_factors/system.h"

namespace csv_export
{
  #define CSV_EXPORT_OK 0
  #define CSV_EXPORT_FILE_ERROR 31

  /// @brief Exports calculated form factors of given scene (mesh_i to mesh_j) to csv file.
  int export_csv(const char* filename, form_factors::scene_t* scene, form_factors::task_t* task);
}