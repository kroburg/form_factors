#include "system.h"
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
      //system = cpu_form_factors::system_create();
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
