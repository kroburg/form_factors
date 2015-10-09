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

#pragma once

#include "../emission/system.h"

namespace malley_emission
{
  struct task_t : emission::task_t
  {
    task_t(int n, float total, float* ws)
      : emission::task_t(n)
      , total_weight(total)
      , weights(ws)
    {}
    /**
    @brief Total weight of all faces (requried to normalize).
    @todo Pass total or 1/total (scale factor) (to use multiplication instead of division in emitted_xxx() functions)?
    */
    float total_weight;

    /**
    @brief Two weights per face: first one in normal direction (frontside), second in opposite (backside).
    @note System will emit at least one ray per face with non zero weight.
    */
    float* weights;
  };

  /// @brief Rays count to be emitted from front side (weight-based).
  int emitted_front(const task_t* task, int face_idx);

  /// @brief Rays count to be emitted from rear side (weight-based).
  int emitted_rear(const task_t* task, int face_idx);

  /// @brief create task for given scene with n_rays rays.
  task_t* task_create(int n_rays, int n_faces);

  /// @brief Free memory for given task
  void task_free(task_t* task);
}

