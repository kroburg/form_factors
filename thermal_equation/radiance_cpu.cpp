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


#include "radiance_cpu.h"
#include "../math/triangle.h"
#include <float.h>
#include <cmath>
#include <limits>
#include <stdlib.h>
#include <cstring>

namespace radiance_equation
{
  const float sigma = 5.670400e-8f;

  /// @brief Extended base system_t (C-style polymorphism)
  struct cpu_system_t : thermal_equation::system_t
  {    
    params_t params;

    subject::scene_t* scene;
    float* face_areas;
    int* face_to_mesh_index;

    // @todo Move form factors calculation of process?
    emission::system_t* emission_calculator;
    ray_caster::scene_t emission_scene;
    emission::task_t* emission_task;
  };

  /// @brief Initializes system with given ray caster after creation.
  int init(cpu_system_t* system, params_t* params)
  {    
    system->params = *params;
    system->face_areas = 0;
    system->face_to_mesh_index = 0;
    system->scene = 0;
    system->emission_calculator = params->emitter;
    system->emission_task = 0;

    return THERMAL_EQUATION_OK;
  }

  /// @brief Shutdowns calculator system prior to free memory.
  int shutdown(cpu_system_t* system)
  {
    system->scene = 0;

    free(system->face_areas);
    system->face_areas = 0;

    free(system->face_to_mesh_index);
    system->face_to_mesh_index = 0;

    system->emission_calculator = 0;
    // @todo Looks like it is better to move task ownership to base system to avoid such code.
    emission::task_free(system->emission_task);
    system->emission_task = 0;

    return THERMAL_EQUATION_OK;
  }

  int set_scene(cpu_system_t* system, subject::scene_t* scene)
  {
    system->scene = scene;
    system->emission_scene = *(ray_caster::scene_t*)scene; // slicing binary compatible scenes

    int r = 0;
    if ((r = emission::system_set_scene(system->emission_calculator, &system->emission_scene)) < 0)
      return r;

    build_faces_areas(system->scene, &system->face_areas);
    build_face_to_mesh_index(scene->n_faces, scene->n_meshes, scene->meshes, &system->face_to_mesh_index);

    emission::task_free(system->emission_task);
    system->emission_task = emission::task_create(system->params.n_rays, scene->n_faces);

    return THERMAL_EQUATION_OK;
  }

  void calculate_weights(const subject::scene_t* scene, const float* face_areas, const float* temperatures, emission::task_t* task)
  {
    // Reset to zero. There may be faces not included in meshes.
    memset(task->weights, 0, scene->n_faces * 2 * sizeof(float));

    float total = 0;
    for (int m = 0; m != scene->n_meshes; ++m)
    { 
      const subject::material_t& material = mesh_material(scene, m);
      const float T = temperatures[m];
      const float density = T;
      const float front_density = density * material.front.emissivity;
      const float rear_density = density * material.rear.emissivity;
     
      const subject::mesh_t& mesh = scene->meshes[m];
      for (int f = mesh.first_idx; f != mesh.first_idx + mesh.n_faces; ++f)
      {
        const float front = front_density * face_areas[f];
        const float rear = rear_density * face_areas[f];
        total += front + rear;
        task->weights[f * 2] = front;
        task->weights[f * 2 + 1] = rear;
      }
    }

    task->total_weight = total;
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
    calculate_weights(system->scene, system->face_areas, task->temperatures, system->emission_task);

    int r = 0;
    if ((r = emission::system_calculate(system->emission_calculator, system->emission_task)) < 0)
      return r;

    // now we have nearest intersection face for every ray in task (if intersection was occurred)
    // calculate form factors between meshes
    const int n_meshes = system->scene->n_meshes;
    
    emission::task_t* emission_task = system->emission_task;
    const ray_caster::task_t* ray_caster_task = emission_task->rays;

    int n_ray = 0;
    for (int m = 0; m != n_meshes; ++m)
    {
      const subject::mesh_t& mesh = system->scene->meshes[m];
      const subject::material_t& material = mesh_material(system->scene, m);
      const float T = task->temperatures[m];
      const float power_density = sigma * (T * T * T * T);
      const float front_density = power_density * material.front.emissivity;
      const float rear_density = power_density * material.rear.emissivity;

      for (int f = mesh.first_idx; f != mesh.first_idx + mesh.n_faces; ++f)
      { 
        const float front_emission = front_density * system->face_areas[f];
        const float rear_emission = rear_density * system->face_areas[f];
        task->emission[m] += front_emission + rear_emission;

        const int face_rays_front = emitted_front(emission_task, f);
        const int face_rays_rear = emitted_rear(emission_task, f);

        const int face_rays = face_rays_front + face_rays_rear;
        for (int j = 0; j != face_rays && n_ray < ray_caster_task->n_tasks; ++j, ++n_ray)
        {
          if (ray_caster_task->hit_face[n_ray])
          {
            const int hit_face_idx = ray_caster_task->hit_face[n_ray] - system->emission_scene.faces;
            const int hit_mesh_idx = face2mesh(system, hit_face_idx);

            // @note Accepting material properties (like absorbance) does not matter. It should handled by emission and ray casting modules.
            const float ray_power = j < face_rays_front ? (front_emission / face_rays_front) : (rear_emission / face_rays_rear);

            task->absorption[hit_mesh_idx] += ray_power; 
          }
        }
      }
    }

    ray_caster::task_free(emission_task->rays);
    emission_task->rays = 0;

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
