#include "cpu.h"

namespace ray_caster
{
  namespace cpu
  {
    struct cpu_system_t : system_t
    {
      scene_t* scene;
    };

    int init(cpu_system_t* system)
    {
      system->scene = 0;
      return RAY_CASTER_OK;
    }

    int shutdown(cpu_system_t* system)
    {
      return RAY_CASTER_OK;
    }

    int set_scene(cpu_system_t* system, scene_t* scene)
    {
      system->scene = scene;
      return RAY_CASTER_OK;
    }

    int prepare(cpu_system_t* system)
    {
      if (system->scene == 0 || system->scene->n_faces == 0)
        return -RAY_CASTER_ERROR;
      return RAY_CASTER_OK;
    }
    
    const system_methods_t methods =
    {
      (int(*)(system_t* system))&init,
      (int(*)(system_t* system))&shutdown,
      (int(*)(system_t* system, ray_caster::scene_t* scene))&set_scene,
      (int(*)(system_t* system))&prepare
    };

    system_t* system_create()
    {
      cpu_system_t* s = (cpu_system_t*)malloc(sizeof(cpu_system_t));
      s->methods = &methods;
      return s;
    }
  }
}
