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
 * This module contains CPU single-threaded implementation of form factors calculator.
 * Calculator is capable to work with CPU or GPU ray caster implementation.
 */

#include "cpu_system.h"
#include "../emission/malley_emission.h"
#include "../ray_caster/system.h"
#include "../math/operations.h"
#include "../math/triangle.h"
#include "../math/mat.h"
#include <float.h>
#include <cmath>
#include <stdlib.h>
#include <cstring>

namespace cpu_form_factors
{
  /// @brief Extended base system_t (C-style polymorphism)
  struct cpu_system_t : form_factors::system_t
  {
    form_factors::scene_t* scene;

    float total_area;
    float* face_weights;

    /// @brief Inverted index to find mesh from face idx.
    int* face_to_mesh;

    emission::system_t* emitter;
    ray_caster::scene_t ray_caster_scene;
  };

  /// @brief Initializes system with given ray caster after creation.
  int init(cpu_system_t* system, emission::system_t* emitter)
  {
    system->scene = 0;
    system->total_area = 0;
    system->face_weights = 0;

    system->face_to_mesh = 0;

    system->emitter = emitter;
    system->ray_caster_scene = { 0, 0 };
   
    return FORM_FACTORS_OK;
  }

  /// @brief Shutdowns calculator system prior to free memory.
  int shutdown(cpu_system_t* system)
  {
    system->scene = 0;

    free(system->face_weights);
    system->face_weights = 0;
    
    free(system->face_to_mesh);
    system->face_to_mesh = 0;

    system->total_area = 0;

    return FORM_FACTORS_OK;
  }

  /// @brief Sets loaded scene (polygons in meshes) for calculator and associated ray caster.
  int set_scene(cpu_system_t* system, form_factors::scene_t* scene)
  {
    system->scene = scene;
    system->ray_caster_scene = { system->scene->n_faces, system->scene->faces };
    
    int r = 0;
    if ((r = emission::system_set_scene(system->emitter, &system->ray_caster_scene)) < 0)
      return r;

    return FORM_FACTORS_OK;
  }

  /// @brief Prepares calculator prior to calculation.
  int prepare(cpu_system_t* system)
  {
    if (system->scene == 0 || system->scene->n_faces == 0 || system->scene->n_meshes == 0)
      return -FORM_FACTORS_EMPTY_SCENE;

    const int n_faces = system->scene->n_faces;

    free(system->face_weights);
    system->face_weights = (float*)malloc(2 * n_faces * sizeof(float));

    // @todo Use subject::build_face_to_mesh_index()
    free(system->face_to_mesh);
    system->face_to_mesh = (int*)malloc(n_faces * sizeof(int));

    // fill face-to-mesh inverted index for every mesh
    // @todo Use subject::build_face_to_mesh_index() function.
    for (int m = 0; m != system->scene->n_meshes; ++m)
    {
      const form_factors::mesh_t& mesh = system->scene->meshes[m];
      const int mesh_n_faces = mesh.n_faces;
      for (int f = 0; f != mesh_n_faces; ++f)
      {
        system->face_to_mesh[mesh.first_idx + f] = m;
      }
    }

    // calculate face weights as face area to total scene area ratio
    for (int i = 0; i != n_faces; ++i)
    {
      form_factors::face_t* face = &system->scene->faces[i];
      float face_area = math::triangle_area(*face);
      system->total_area += 2.f * face_area;
      system->face_weights[2 * i] = face_area;
      system->face_weights[2 * i + 1] = face_area;
    }

    // check scene's total area
    if (system->total_area < FLT_EPSILON)
      return -FORM_FACTORS_SCENE_TOO_SMALL;
    
    return FORM_FACTORS_OK;
  }

  /// @brief Returns mesh for given face.
  int face2mesh(cpu_system_t* system, int face_idx)
  {
    return system->face_to_mesh[face_idx];
  }

  /**
   *  @brief Calculates form factors for given system.
   *
   *  System uses ray caster (@see init()) and given task for N rays and scene's meshes.
   */
  int calculate(cpu_system_t* system, form_factors::task_t* task)
  {
    malley_emission::task_t emission_task(task->n_rays, system->total_area, system->face_weights);
    
    int r = 0;
    if ((r = emission::system_calculate(system->emitter, &emission_task)) < 0)
    {
      ray_caster::task_free(emission_task.rays);
      return r;
    }

    // now we have nearest intersection face for every ray in task (if intersection was occurred)
    // calculate form factors between meshes
    const int n_meshes = system->scene->n_meshes;
    memset(task->form_factors, 0, n_meshes * n_meshes * sizeof(float));

    ray_caster::task_t* ray_caster_task = emission_task.rays;
    
    int n_ray = 0;
    for (int m = 0; m != n_meshes; ++m)
    { 
      int mesh_outgoing_rays = 0;

      const form_factors::mesh_t& mesh = system->scene->meshes[m];
      const int mesh_n_faces = mesh.n_faces;
      for (int f = 0; f != mesh_n_faces; ++f)
      {
        const int face_idx = mesh.first_idx + f;
        const ray_caster::face_t& face = system->scene->faces[face_idx];
        const int face_rays_front = emitted_front(&emission_task, face_idx);
        const int face_rays_rear = emitted_rear(&emission_task, face_idx);
        const int face_rays = face_rays_front + face_rays_rear;
        for (int j = 0; j != face_rays && n_ray < emission_task.rays->n_tasks; ++j, ++n_ray)
        {
          ++mesh_outgoing_rays;
          if (ray_caster_task->hit_face[n_ray])
          {
            int hit_face_idx = ray_caster_task->hit_face[n_ray] - system->ray_caster_scene.faces;
            int hit_mesh_idx = face2mesh(system, hit_face_idx);
            task->form_factors[m * n_meshes + hit_mesh_idx] += 1.f;
          }
        }
      }

      float mesh_ratio = 1.f / (float)mesh_outgoing_rays;
      for (int j = 0; j != n_meshes; ++j)
        task->form_factors[m * n_meshes + j] *= mesh_ratio;
    }

    ray_caster::task_free(emission_task.rays);
    
    return FORM_FACTORS_OK;
  }

  /// @brief Creates virtual methods table from local methods.
  const form_factors::system_methods_t methods =
  {
    (int(*)(form_factors::system_t* system, emission::system_t* emitter))&init,
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
