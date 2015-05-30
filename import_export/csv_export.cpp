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

#include "csv_export.h"
#include <fstream>

namespace csv_export
{
  int export_csv(const char* filename, form_factors::scene_t* scene, form_factors::task_t* task)
  {
    std::ofstream out(filename);
    if (!out.is_open())
      return -CSV_EXPORT_FILE_ERROR;

    const int n_meshes = scene->n_meshes;
    for (int i = 0; i != n_meshes; ++i)
    {
      for (int j = 0; j != n_meshes; ++j)
      {
        if (j != 0)
          out << ";";
        const float factor = task->form_factors[i * n_meshes + j];
        out << factor;
      }
      out << std::endl;
    }
    out.close();

    return CSV_EXPORT_OK;
  }
}