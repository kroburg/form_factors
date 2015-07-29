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
#include "../ray_caster/system.h"
#include "../math/operations.h"
#include "../math/triangle.h"
#include "../math/mat.h"
#include <float.h>
#include <cmath>
#include <limits>
#include <stdlib.h>
#include <cstring>

namespace cpu_form_factors
{
  /// @brief Extended base system_t (C-style polymorphism)
  struct cpu_system_t : form_factors::system_t
  {
    form_factors::scene_t* scene;

    int n_faces;
    face_t* faces;
    float total_area;

    /// @brief Inverted index to find mesh from face idx.
    int* face_to_mesh;

    emission::system_t* emitter;
    ray_caster::scene_t ray_caster_scene;
  };

  /// @brief Initializes system with given ray caster after creation.
  int init(cpu_system_t* system, emission::system_t* emitter)
  {
    system->scene = 0;
    system->n_faces = 0;
    system->faces = 0;
    system->total_area = 0;

    system->face_to_mesh = 0;

    system->emitter = emitter;
    system->ray_caster_scene = { 0, 0 };
   
    return FORM_FACTORS_OK;
  }

  /// @brief Shutdowns calculator system prior to free memory.
  int shutdown(cpu_system_t* system)
  {
    system->scene = 0;
    system->n_faces = 0;
    
    free(system->face_to_mesh);
    system->face_to_mesh = 0;

    free(system->faces);
    system->faces = 0;
    system->total_area = 0;

    return FORM_FACTORS_OK;
  }

  /// @brief Sets loaded scene (polygons in meshes) for calculator and associated ray caster.
  int set_scene(cpu_system_t* system, form_factors::scene_t* scene)
  {
    system->scene = scene;
    system->n_faces = scene->n_faces;

    system->ray_caster_scene = { system->scene->n_faces, system->scene->faces };
    
    int r = 0;
    if ((r = emission::system_set_scene(system->emitter, &system->ray_caster_scene)) < 0)
      return r;

    return FORM_FACTORS_OK;
  }

  /// @brief Calculates area for whole scene.
  float calculate_area(form_factors::scene_t* scene)
  {
    float result = 0;
    for (int i = 0; i != scene->n_faces; ++i)
    {
      result += triangle_area(scene->faces[i]);
    }
    return result;
  }


  /// @brief Prepares calculator prior to calculation.
  int prepare(cpu_system_t* system)
  {
    if (system->scene == 0 || system->scene->n_faces == 0 || system->scene->n_meshes == 0)
      return -FORM_FACTORS_ERROR;

    // prepare ray caster
    int r = 0;
    if ((r = ray_caster::system_prepare(system->ray_caster)) < 0)
      return r;

    // prepare system faces
    free(system->faces);
    system->faces = (face_t*)malloc(system->n_faces * sizeof(face_t));

    free(system->face_to_mesh);
    system->face_to_mesh = (int*)malloc(system->n_faces * sizeof(int));

    // fill face-to-mesh inverted index for every mesh
    for (int m = 0; m != system->scene->n_meshes; ++m)
    {
      const form_factors::mesh_t& mesh = system->scene->meshes[m];
      const int mesh_n_faces = mesh.n_faces;
      for (int f = 0; f != mesh_n_faces; ++f)
      {
        system->face_to_mesh[mesh.first_idx + f] = m;
      }
    }

    // check scene's total area
    system->total_area = calculate_area(system->scene);
    if (system->total_area < FLT_EPSILON)
      return -FORM_FACTORS_ERROR;

    // calculation of every face area to total scene area ratio
    for (int i = 0; i != system->n_faces; ++i)
    {
      form_factors::face_t* source_face = &system->scene->faces[i];
      float face_area = math::triangle_area(*source_face);
      float face_weight = face_area / system->total_area;

      // copy to own faces array
      /// @todo: Why another copy of scene's faces?
      face_t* target_face = &system->faces[i];
      *((math::triangle_t*)target_face) = *source_face;
      target_face->weight = face_weight;
    }
    
    return FORM_FACTORS_OK;
  }

  /// @brief Minimum number of rays per scene.
  int calculate_n_rays(cpu_system_t* system, int rays_requested)
  {
    int result = 0;
    for (int i = 0; i != system->n_faces; ++i)
    {
      int faceRaysCount = std::max<int>(1, (int)(rays_requested * system->faces[i].weight));
      result += faceRaysCount;
    }
    return result;
  }

  /// @brief Generates uniformly distributed points on triangle.
  math::vec3 pick_face_point(cpu_system_t* system, const face_t& face)
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
  math::mat33 pick_face_rotation(const face_t& face, math::vec3 z)
  {
    math::vec3 v0 = face.points[1] - face.points[0];
    math::vec3 v1 = face.points[2] - face.points[0];
    math::vec3 norm = cross(v0, v1);
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
  ray_caster::task_t* make_caster_task(cpu_system_t* system, int n_rays)
  {
    ray_caster::task_t* task = ray_caster::task_create(n_rays);
    const int n_meshes = system->scene->n_meshes;
    int n_ray = 0;

    // For every mesh in scene
    for (int m = 0; m != n_meshes; ++m)
    {
      const form_factors::mesh_t& mesh = system->scene->meshes[m];
      const int mesh_n_faces = mesh.n_faces;
      for (int f = 0; f != mesh_n_faces; ++f)
      {
        const face_t& face = system->faces[mesh.first_idx + f];

        // For given face of given mesh number of rays is proportional to
        // ratio of face's area to whole scene area (weight).
        const int face_rays = std::max<int>(1, (int)(n_rays * face.weight));

        // Store rotation for for given face (from Z axis towards face's normal).
        math::mat33 rotation = pick_face_rotation(face, math::make_vec3(0, 0, 1));

        for (int j = 0; j != face_rays && n_ray < n_rays; ++j, ++n_ray)
        {
          // Take reference to ray being generated
          ray_caster::ray_t& ray = task->ray[n_ray];

          // Randomly generated ray's origin on the face
          math::vec3 origin = pick_face_point(system, face);

          // Pick direction from cosine-weighted distribution
          math::vec3 malley = pick_malley_point(system);

          if (j > face_rays / 2) {
            // One half rays from front side and one half from back
            malley.z = -malley.z;
          }

          // Rotate ray towards face's normal
          math::vec3 relative_dist = rotation * malley;

          // Store by reference
          ray = { origin + relative_dist * 0.0001f, origin + relative_dist };
        }
      }
    }

    return task;
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
    int r = 0;

    // perform ray casting
    int n_rays = calculate_n_rays(system, task->n_rays);
    ray_caster::task_t* ray_caster_task = make_caster_task(system, n_rays);
    if ((r = ray_caster::system_cast(system->ray_caster, ray_caster_task)) < 0)
      return r;

    // now we have nearest intersection face for every ray in task (if intersection was occurred)
    // calculate form factors between meshes
    const int n_meshes = system->scene->n_meshes;
    memset(task->form_factors, 0, n_meshes * n_meshes * sizeof(float));
    
    int n_ray = 0;
    for (int m = 0; m != n_meshes; ++m)
    { 
      int mesh_outgoing_rays = 0;

      const form_factors::mesh_t& mesh = system->scene->meshes[m];
      const int mesh_n_faces = mesh.n_faces;
      for (int f = 0; f != mesh_n_faces; ++f)
      {
        const face_t& face = system->faces[mesh.first_idx + f];
        const int face_rays = std::max<int>(1, (int)(n_rays * face.weight));
        for (int j = 0; j != face_rays && n_ray < n_rays; ++j, ++n_ray)
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

    ray_caster::task_free(ray_caster_task);
    
    return FORM_FACTORS_OK;
  }

  /// @brief Creates virtual methods table from local methods.
  const form_factors::system_methods_t methods =
  {
    (int(*)(form_factors::system_t* system, ray_caster::system_t* ray_caster))&init,
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
