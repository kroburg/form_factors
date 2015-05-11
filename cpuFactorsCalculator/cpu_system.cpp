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

#include "cpu_system.h"
#include "../ray_caster/system.h"
#include "../math/operations.h"
#include "../math/triangle.h"
#include "../math/mat.h"
#include <float.h>
#include <random>
#include <cmath>
#include <limits>
#include <stdlib.h>

#define M_2PI           6.28318530717958647692528676655900576

namespace cpu_form_factors
{
  struct cpu_system_t : form_factors::system_t
  {
    form_factors::scene_t* scene;
    int n_faces;
    face_t* faces;
    float total_area;

    int* face_to_mesh;

    ray_caster::system_t* ray_caster;
    ray_caster::scene_t ray_caster_scene;

    std::mt19937 TPGenA;
    std::mt19937 TPGenB;
    std::mt19937 HSGenTheta;
    std::mt19937 HSGenR;
    std::uniform_real_distribution<float> Distr_0_1;
    std::uniform_real_distribution<float> Distr_0_2PI;
  };

  int init(cpu_system_t* system, ray_caster::system_t* ray_caster)
  {
    system->scene = 0;
    system->n_faces = 0;
    system->faces = 0;
    system->total_area = 0;

    system->face_to_mesh = 0;

    system->ray_caster = ray_caster;
    system->ray_caster_scene = { 0, 0 };

    system->TPGenA = std::mt19937(1);
    system->TPGenB = std::mt19937(2);
    system->HSGenTheta = std::mt19937(3);
    system->HSGenR = std::mt19937(4);
    system->Distr_0_1 = std::uniform_real_distribution<float>(0, 1);
    system->Distr_0_2PI = std::uniform_real_distribution<float>(0, float(M_2PI));
    return FORM_FACTORS_OK;
  }

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

  int set_scene(cpu_system_t* system, form_factors::scene_t* scene)
  {
    system->scene = scene;
    system->n_faces = scene->n_faces;

    system->ray_caster_scene = { system->scene->n_faces, system->scene->faces };
    
    int r = 0;
    if ((r = ray_caster::system_set_scene(system->ray_caster, &system->ray_caster_scene)) < 0)
      return r;

    return FORM_FACTORS_OK;
  }

  float calculate_area(form_factors::scene_t* scene)
  {
    float result = 0;
    for (int i = 0; i != scene->n_faces; ++i)
    {
      result += triangle_area(scene->faces[i]);
    }
    return result;
  }

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

    for (int m = 0; m != system->scene->n_meshes; ++m)
    {
      const form_factors::mesh_t& mesh = system->scene->meshes[m];
      const int mesh_n_faces = mesh.n_faces;
      for (int f = 0; f != mesh_n_faces; ++f)
      {
        system->face_to_mesh[mesh.first_idx + f] = m;
      }
    }

    system->total_area = calculate_area(system->scene);
    if (system->total_area < FLT_EPSILON)
      return -FORM_FACTORS_ERROR;

    for (int i = 0; i != system->n_faces; ++i)
    {
      form_factors::face_t* source_face = &system->scene->faces[i];
      float face_area = math::triangle_area(*source_face);
      float face_weight = face_area / system->total_area;
      face_t* target_face = &system->faces[i];
      *((math::triangle_t*)target_face) = *source_face;
      target_face->weight = face_weight;
    }
    
    return FORM_FACTORS_OK;
  }

  int calculate_n_rays(cpu_system_t* system, int rays_requested)
  {
    int result = 0;
    for (int i = 0; i != system->n_faces; ++i)
    {
      int faceRaysCount = (int)(rays_requested * system->faces[i].weight);
      result += faceRaysCount;
    }
    return result;
  }

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
    return a * v0 + b * v1;
  }

  math::mat33 pick_face_rotation(const face_t& face)
  {
    math::vec3 v0 = face.points[1] - face.points[0];
    math::vec3 v1 = face.points[2] - face.points[0];
    math::vec3 norm = cross(v0, v1);
    math::vec3 z = math::make_vec3(0, 0, 1);
    return math::rotate_towards(z, norm);
  }

  math::vec3 pick_malley_point(cpu_system_t* system)
  {
    float r = system->Distr_0_1(system->HSGenR);
    float rad = sqrtf(r);
    float phi = system->Distr_0_2PI(system->HSGenTheta);
    return{ rad * cosf(phi), rad * sinf(phi), sqrtf(1 - r) };
  }

  ray_caster::task_t* make_caster_task(cpu_system_t* system, int n_rays)
  {
    ray_caster::task_t* task = (ray_caster::task_t*)malloc(sizeof(ray_caster::task_t));
    task->n_tasks = n_rays;
    task->ray = (ray_caster::ray_t*)malloc(n_rays * sizeof(ray_caster::ray_t));
    task->hit_face = (ray_caster::face_t**)malloc(n_rays * sizeof(ray_caster::face_t*));
    task->hit_point = (math::vec3*)malloc(n_rays * sizeof(math::vec3));

    const int n_meshes = system->scene->n_meshes;
    int n_ray = 0;
    for (int m = 0; m != n_meshes; ++m)
    {
      const form_factors::mesh_t& mesh = system->scene->meshes[m];
      const int mesh_n_faces = mesh.n_faces;
      for (int f = 0; f != mesh_n_faces; ++f)
      {
        const face_t& face = system->faces[mesh.first_idx + f];
        const int face_rays = (int)(n_rays * face.weight);
        math::mat33 rotation = pick_face_rotation(face);

        for (int j = 0; j != face_rays && n_ray < n_rays; ++j, ++n_ray)
        {
          ray_caster::ray_t& ray = task->ray[n_ray];
          math::vec3 origin = pick_face_point(system, face);
          math::vec3 relative_dist = rotation * pick_malley_point(system);
          ray = { origin + relative_dist * 0.0001f, origin + relative_dist };
        }
      }

    }

    return task;
  }

  int face2mesh(cpu_system_t* system, int face_idx)
  {
    return system->face_to_mesh[face_idx];
  }

  int calculate(cpu_system_t* system, form_factors::task_t* task)
  {
    int r = 0;

    // perform ray casting
    int n_rays = calculate_n_rays(system, task->n_rays);
    ray_caster::task_t* ray_caster_task = make_caster_task(system, n_rays);
    if ((r = ray_caster::system_cast(system->ray_caster, ray_caster_task)) < 0)
      return r;
    

    // calculate meshes form factors
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
        const int face_rays = (int)(n_rays * face.weight);
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
    
    return -FORM_FACTORS_ERROR;
  }

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
