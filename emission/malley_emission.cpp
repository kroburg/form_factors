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
*  Malley emission support.
*/

#include "malley_emission.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace malley_emission
{
  task_t* task_create(int n_rays, int n_faces)
  {
    task_t* task = (task_t*)malloc(sizeof(task_t));
    task->n_rays = n_rays;
    task->total_weight = 0;
    const int weights_mem_size = 2 * n_faces * sizeof(float);
    task->weights = (float*)malloc(weights_mem_size);
    memset(task->weights, 0, weights_mem_size);
    task->rays = 0;
    return task;
  }

  void task_free(task_t* task)
  {
    if (task)
    {
      free(task->weights);
      ray_caster::task_free(task->rays);
      free(task);
    }
  }

  int emitted_front(const task_t* task, int face_idx)
  {
    const float weight = task->weights[2 * face_idx];
    return weight > 0 ? std::max<int>(1, (int)(task->n_rays * (weight / task->total_weight))) : 0;
  }

  int emitted_rear(const task_t* task, int face_idx)
  {
    const float weight = task->weights[2 * face_idx + 1];
    return weight > 0 ? std::max<int>(1, (int)(task->n_rays * (weight / task->total_weight))) : 0;
  }
}
