#include "cpu_system.h"
#include "../math/operations.h"
#include "../math/triangle.h"
#include <limits>
#include <stdlib.h>

namespace cpu_ray_caster
{
  struct cpu_system_t : ray_caster::system_t
  {
    ray_caster::scene_t* scene;
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

  int set_scene(cpu_system_t* system, ray_caster::scene_t* scene)
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

  int cast(cpu_system_t* system, ray_caster::task_t* task)
  {
    using namespace ray_caster;
    for (int t = 0; t != task->n_tasks; ++t)
    {
      math::ray_t ray = task->ray[t];
      math::point_t min_distance = std::numeric_limits<math::point_t>::max();
      task->hit_face[t] = 0;
      for (int f = 0; f != system->scene->n_faces; ++f)
      {
        math::triangle_t triangle = system->scene->faces[f];
        math::vec3 point;
        int check_result = triangle_intersect(ray, triangle, &point);
        if (check_result == TRIANGLE_INTERSECTION_UNIQUE)
        {
          math::vec3 space_distance = point - ray.origin;
          math::point_t new_distance = dot(space_distance, space_distance);
          if (new_distance < min_distance)
          {
            min_distance = new_distance;
            task->hit_face[t] = &system->scene->faces[f];
            task->hit_point[t] = point;
          }
        }
      }
    }

    return RAY_CASTER_OK;
  }

  const ray_caster::system_methods_t methods =
  {
    (int(*)(ray_caster::system_t* system))&init,
    (int(*)(ray_caster::system_t* system))&shutdown,
    (int(*)(ray_caster::system_t* system, ray_caster::scene_t* scene))&set_scene,
    (int(*)(ray_caster::system_t* system))&prepare,
    (int(*)(ray_caster::system_t* system, ray_caster::task_t* task))&cast,
  };

  ray_caster::system_t* system_create()
  {
    cpu_system_t* s = (cpu_system_t*)malloc(sizeof(cpu_system_t));
    s->methods = &methods;
    return s;
  }
}
