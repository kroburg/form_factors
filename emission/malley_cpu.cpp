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
* This module contains CPU single-threaded implementation of emission calculator.
* Calculator is capable of work with CPU or GPU ray caster implementation.
*/

#include "malley_cpu.h"
#include "malley_emission.h"
#include "../ray_caster/system.h"
#include "../math/operations.h"
#include "../math/triangle.h"
#include "../math/mat.h"
#include <float.h>
#include <random>
#include <cmath>
#include <limits>
#include <stdlib.h>
#include <cstring>

namespace malley_cpu
{
  /// @brief Extended base system_t (C-style polymorphism)
  struct cpu_system_t : emission::system_t
  {
    ray_caster::scene_t* scene;
    ray_caster::system_t* ray_caster;

    // Mersenne's twister uniformly distributed [0, 1) generators.
    std::mt19937 TPGenA;
    std::mt19937 TPGenB;
    std::mt19937 HSGenTheta;
    std::mt19937 HSGenR;

    // [0, 1) and [0, 2 * PI) redistributions
    std::uniform_real_distribution<float> Distr_0_1;
    std::uniform_real_distribution<float> Distr_0_2PI;
  };

  /// @brief Initializes system with given ray caster after creation.
  int init(cpu_system_t* system, ray_caster::system_t* ray_caster)
  {
    system->scene = 0;
    system->ray_caster = ray_caster;

    system->TPGenA = std::mt19937(1);
    system->TPGenB = std::mt19937(2);
    system->HSGenTheta = std::mt19937(3);
    system->HSGenR = std::mt19937(4);
    system->Distr_0_1 = std::uniform_real_distribution<float>(0, 1);
    system->Distr_0_2PI = std::uniform_real_distribution<float>(0, float(M_2PI));
    return EMISSION_OK;
  }

  /// @brief Shutdowns calculator system prior to free memory.
  int shutdown(cpu_system_t* system)
  {
    system->scene = 0;
    return EMISSION_OK;
  }

  /// @brief Sets loaded scene (polygons in meshes) for calculator and associated ray caster.
  int set_scene(cpu_system_t* system, ray_caster::scene_t* scene)
  {
    if (scene == 0 || scene->n_faces == 0)
      return -EMISSION_ERROR;

    system->scene = scene;
    
    int r = 0;
    if ((r = ray_caster::system_set_scene(system->ray_caster, system->scene)) < 0)
      return r;

    if ((r = ray_caster::system_prepare(system->ray_caster)) < 0)
      return r;

    return EMISSION_OK;
  }

  

  int calculate_n_rays(malley_emission::task_t* task, int n_faces)
  {
    int result = 0;
    for (int f = 0; f != n_faces; ++f)
    {
      const int face_rays_front = emitted_front(task, f);
      const int face_rays_rear = emitted_rear(task, f);
      result += face_rays_front + face_rays_rear;
    }
    return result;
  }

  /// @brief Generates uniformly distributed points on triangle.
  math::vec3 pick_face_point(cpu_system_t* system, const math::face_t& face)
  {
    float a = system->Distr_0_1(system->TPGenA);
    float b = system->Distr_0_1(system->TPGenB);
    if (a + b > 1)
    {
      a = 1.f - a;
      b = 1.f - b;
    }

    math::vec3 v0 = face.points[1] - face.points[0];
    math::vec3 v1 = face.points[2] - face.points[0];
    return face.points[0] + a * v0 + b * v1;
  }

  /// @brief Creates rotation matrix of z vector towards face's normal.
  math::mat33 pick_face_rotation(const math::face_t& face, math::vec3 z)
  { 
    math::vec3 norm = triangle_normal(face);
    return math::rotate_towards(z, norm);
  }

  /// @brief Generates cosine-weighted distribution of points on hemisphere
  /// with radius of 1 and normal of (0, 0, 1).
  math::vec3 pick_malley_point(cpu_system_t* system)
  {
    float r = system->Distr_0_1(system->HSGenR);
    float rad = sqrtf(r);
    float phi = system->Distr_0_2PI(system->HSGenTheta);
    return { rad * cosf(phi), rad * sinf(phi), sqrtf(1 - r) };
  }

  /// @brief Creates task with n_rays random generated rays.
  ray_caster::task_t* make_caster_task(cpu_system_t* system, malley_emission::task_t* task)
  { 
    int n_real_rays = calculate_n_rays(task, system->scene->n_faces);
    if (n_real_rays == 0)
      return 0;
    ray_caster::task_t* ray_caster_task = ray_caster::task_create(n_real_rays);
    ray_caster::scene_t* scene = system->scene;
    const float* weights = task->weights;
    int n_ray = 0;
    
    for (int f = 0; f != scene->n_faces; ++f)
    {
      const ray_caster::face_t& face = system->scene->faces[f];

      // For given face number of rays is proportional to faces front/back weights.
      const int face_rays_front = emitted_front(task, f);
      const int face_rays_rear = emitted_rear(task, f);
      const int face_rays = face_rays_front + face_rays_rear;

      // Store rotation for for given face (from Z axis towards face's normal).
      math::mat33 rotation = pick_face_rotation(face, math::make_vec3(0, 0, 1));

      for (int j = 0; j != face_rays && n_ray < n_real_rays; ++j, ++n_ray)
      {
        // Take reference to ray being generated
        ray_caster::ray_t& ray = ray_caster_task->ray[n_ray];

        // Randomly generated ray's origin on the face
        math::vec3 origin = pick_face_point(system, face);

        // Pick direction from cosine-weighted distribution
        math::vec3 malley = pick_malley_point(system);

        if (j >= face_rays_front) {
          // One half rays from front side and one half from back
          malley.z = -malley.z;
        }

        // Rotate ray towards face's normal
        math::vec3 relative_dist = rotation * malley;

        // Store by reference
        ray = { origin + relative_dist * 0.0001f, origin + relative_dist };
      }
    }
    
    return ray_caster_task;
  }

  /**
   *  @brief Calculates emission for given system.
   *  @detail System uses ray caster (@see init()) and given task for N rays and scene's faces.
   */
  int calculate(cpu_system_t* system, malley_emission::task_t* task)
  {
    if (task->rays)
    {
      // @todo Provide realloc
      ray_caster::task_free(task->rays);
      task->rays = 0;
    }
    task->rays = make_caster_task(system, task);
    if (task->rays == 0)
      return EMISSION_OK;
    int r = 0;
    if ((r = ray_caster::system_cast(system->ray_caster, task->rays)) < 0)
      return r;

    return EMISSION_OK;
  }

  /// @brief Creates virtual methods table from local methods.
  const emission::system_methods_t methods =
  {
    (int(*)(emission::system_t* system, ray_caster::system_t* ray_caster))&init,
    (int(*)(emission::system_t* system))&shutdown,
    (int(*)(emission::system_t* system, ray_caster::scene_t* scene))&set_scene,
    (int(*)(emission::system_t* system, emission::task_t* task))&calculate,
  };

  emission::system_t* system_create()
  {
    cpu_system_t* s = (cpu_system_t*)malloc(sizeof(cpu_system_t));
    s->methods = &methods;
    return s;
  }
}
