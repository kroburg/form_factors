// Copyright 2015 Stepan Tezyunichev (stepan.tezyunichev@gmail.com).
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

#include "cpu_system.h"
#include "../math/operations.h"
#include "../math/triangle.h"
#include <limits>
#include <stdlib.h>

namespace cpu_form_factors
{
  struct cpu_system_t : form_factors::system_t
  {
    form_factors::scene_t* scene;
  };

  int init(cpu_system_t* system)
  {
    system->scene = 0;
    return FORM_FACTORS_OK;
  }

  int shutdown(cpu_system_t* system)
  {
    return FORM_FACTORS_OK;
  }

  int set_scene(cpu_system_t* system, form_factors::scene_t* scene)
  {
    system->scene = scene;
    return FORM_FACTORS_OK;
  }

  int prepare(cpu_system_t* system)
  {
    if (system->scene == 0 || system->scene->n_faces == 0 || system->scene->n_meshes == 0)
      return -FORM_FACTORS_ERROR;
    
    return FORM_FACTORS_OK;
  }

  int calculate(cpu_system_t* system, form_factors::task_t* task)
  {
    return -FORM_FACTORS_ERROR;
  }

  const form_factors::system_methods_t methods =
  {
    (int(*)(form_factors::system_t* system))&init,
    (int(*)(form_factors::system_t* system))&shutdown,
    (int(*)(form_factors::system_t* system, form_factors::scene_t* scene))&set_scene,
    (int(*)(form_factors::system_t* system))&prepare,
    (int(*)(form_factors::system_t* system, form_factors::task_t* task))&calculate,
  };

  form_factors::system_t* system_create()
  {
    cpu_system_t* s = (cpu_system_t*)malloc(sizeof(cpu_system_t));
    s->methods = &methods;
    return s;
  }
}
