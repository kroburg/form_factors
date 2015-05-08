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

#include "system.h"
#include "../cpuFactorsCalculator/cpu_system.h"
#include <stdlib.h>

namespace form_factors
{
  scene_t* scene_create()
  {
    scene_t* s = (scene_t*)malloc(sizeof(scene_t));
    *s = { 0, 0 };
    return s;
  }

  void scene_free(scene_t* scene)
  {
    if (scene)
    {
      free(scene->faces);
      free(scene->meshes);
    }
    free(scene);
  }

  system_t* system_create(int type)
  {
    system_t* system = 0;
    switch (type)
    {
    case FORM_FACTORS_CPU:
      system = cpu_form_factors::system_create();
      break;

    default:
      return 0;
    }

    system_init(system);

    return system;
  }

  system_t* system_create_default()
  {
    return system_create(FORM_FACTORS_CPU);
  }

  void system_free(system_t* system)
  {
    system_shutdown(system);
    free(system);
  }

  int system_init(system_t* system)
  {
    return system->methods->init(system);
  }

  int system_shutdown(system_t* system)
  {
    return system->methods->shutdown(system);
  }

  int system_set_scene(system_t* system, scene_t* scene)
  {
    return system->methods->set_scene(system, scene);
  }

  int system_prepare(system_t* system)
  {
    return system->methods->prepare(system);
  }

  int system_calculate(system_t* system, task_t* task)
  {
    return system->methods->calculate(system, task);
  }
}
