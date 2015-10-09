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


#include "parallel_rays_cpu.h"
#include "../emission/parallel_rays_cpu.h"
#include "../math/operations.h"
#include "../math/triangle.h"
#include "../math/mat.h"

namespace parallel_rays_cpu
{
  /// @brief Extended base system_t (C-style polymorphism)
  struct cpu_system_t : thermal_equation::system_t
  {
    params_t params;

    subject::scene_t* scene;
    int* face_to_mesh_index;
    math::sphere_t bsphere;

    // @todo Move form factors calculation of process?
    emission::system_t* emission_calculator;
    ray_caster::scene_t emission_scene;
    parallel_rays_emission_cpu::task_t emission_task;
  };

  /// @brief Initializes system with given ray caster after creation.
  int init(cpu_system_t* system, params_t* params)
  {
    system->params = *params;
    system->face_to_mesh_index = 0;
    system->scene = 0;
    system->emission_calculator = params->emitter;
    system->emission_task.n_rays = params->n_rays;
    system->emission_task.rays = 0;

    return THERMAL_EQUATION_OK;
  }

  /// @brief Shutdowns calculator system prior to free memory.
  int shutdown(cpu_system_t* system)
  {
    system->scene = 0;

    free(system->face_to_mesh_index);
    system->face_to_mesh_index = 0;

    system->emission_calculator = 0;
    ray_caster::task_free(system->emission_task.rays);

    return THERMAL_EQUATION_OK;
  }

  int set_scene(cpu_system_t* system, subject::scene_t* scene)
  {
    system->scene = scene;
    system->emission_scene = *(ray_caster::scene_t*)scene; // slicing binary compatible scenes

    int r = 0;
    if ((r = emission::system_set_scene(system->emission_calculator, &system->emission_scene)) < 0)
      return r;

    build_face_to_mesh_index(scene->n_faces, scene->n_meshes, scene->meshes, &system->face_to_mesh_index);

    system->bsphere = math::triangles_bsphere(scene->faces, scene->n_faces);

    return THERMAL_EQUATION_OK;
  }

  int face2mesh(cpu_system_t* system, int face_idx)
  {
    return system->face_to_mesh_index[face_idx];
  }

  /**
  *  @brief Calculates thermal flow for a given scene.
  */
  int calculate(cpu_system_t* system, thermal_equation::task_t* task)
  {
    if (system->params.source == 0)
      return THERMAL_EQUATION_OK;

    source_t source = system->params.source(system->params.source_param);
    float radius = system->bsphere.radius * 1.1f;
    system->emission_task.distance = 1.1f + radius;
    system->emission_task.direction = source.direction;
    system->emission_task.height = radius * 2;
    system->emission_task.width = radius * 2;

    int r = 0;
    if ((r = emission::system_calculate(system->emission_calculator, &system->emission_task)) < 0)
      return r;

    const ray_caster::task_t* ray_caster_task = system->emission_task.rays;

    if (ray_caster_task == 0)
      return THERMAL_EQUATION_OK;

     float ray_power = source.power * (4 * radius * radius) / system->params.n_rays;

    for (int r = 0; r != ray_caster_task->n_tasks; ++r)
    {
      if (ray_caster_task->hit_face[r])
      {
        math::vec3 normale = math::triangle_normal(*ray_caster_task->hit_face[r]);
        float side = dot(normale, source.direction);
        int mesh_idx = face2mesh(system, ray_caster_task->hit_face[r] - system->emission_scene.faces);
        const subject::material_t& material = mesh_material(system->scene, mesh_idx);
        task->absorption[mesh_idx] += (side < 0 ? material.front.absorbance : material.rear.absorbance) * ray_power;
      }
    }

    ray_caster::task_free(system->emission_task.rays);
    system->emission_task.rays = 0;

    return THERMAL_EQUATION_OK;
  }

  /// @brief Creates virtual methods table from local methods.
  const thermal_equation::system_methods_t methods =
  {
    (int(*)(thermal_equation::system_t* system, void* params))&init,
    (int(*)(thermal_equation::system_t* system))&shutdown,
    (int(*)(thermal_equation::system_t* system, subject::scene_t* scene))&set_scene,
    (int(*)(thermal_equation::system_t* system, thermal_equation::task_t* task))&calculate,
  };

  thermal_equation::system_t* system_create()
  {
    cpu_system_t* s = (cpu_system_t*)malloc(sizeof(cpu_system_t));
    s->methods = &methods;
    return s;
  }
}
