#include "ray_caster.cuh"

#include <helper_cuda.h>

namespace ray_caster
{
  namespace cuda
  {
    struct cuda_system_t : system_t
    {
      scene_t* scene;
      int dev_id;
    };

    int init(cuda_system_t* system)
    {
      system->scene = 0;
      const char* argv[] = { "" };
      system->dev_id = findCudaDevice(1, (const char **)argv);
      return RAY_CASTER_OK;
    }

    int shutdown(cuda_system_t* system)
    {
      cudaDeviceReset();
      return RAY_CASTER_OK;
    }

    int set_scene(cuda_system_t* system, scene_t* scene)
    {
      system->scene = scene;
      return RAY_CASTER_OK;
    }

    int prepare(cuda_system_t* system)
    {
      if (system->scene == 0)
        return RAY_CASTER_ERROR;
      return RAY_CASTER_OK;
    }

    const system_methods_t methods =
    {
      (int(*)(system_t* system))&init,
      (int(*)(system_t* system))&shutdown,
      (int(*)(system_t* system, scene_t* scene))&set_scene,
      (int(*)(system_t* system))&prepare
    };

    system_t* system_create()
    {
      cuda_system_t* s = (cuda_system_t*)malloc(sizeof(cuda_system_t));
      s->methods = &methods;
      return s;
    }
  }
}