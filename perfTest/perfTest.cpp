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
 * This module contains performance test console utility for calculator (CPU) and ray caster (CPU, GPU).
 */

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <cmath>

#include "../ray_caster/system.h"
#include "../form_factors/system.h"
#include "../math/operations.h"
#include "../thermal_solution/system.h"
#include "../thermal_equation/system.h"
#include "../thermal_equation/radiance_cpu.h"
#include "../subject/generator.h"
#include "../import_export/obj_export.h"

#include <helper_timer.h>

using namespace ray_caster;

/// @brief Generates random triangles with vertices located on sphere's surface (confette scene).
scene_t* MakeConfettiScene(int n_faces, float radius, subject::generator_t* generator)
{
  face_t* faces = (face_t*)malloc(n_faces * sizeof(face_t));
  scene_t* scene = (scene_t*)malloc(sizeof(scene_t));
  *scene = { n_faces, faces };
  for (int i = 0; i != n_faces; ++i)
  {
    face_t& f = faces[i];
    generator_surface_point(generator, 3, f.points);
    math::vec3 position;
    generator_surface_point(generator, 1, &position);
    position *= radius;
    f.points[0] += position;
    f.points[1] += position;
    f.points[2] += position;
  }  
  
  return scene;
}

/// @brief Creates task with n_rays rays from sphere's surface towards center.
task_t* MakeCollapsingRays(int n_rays, float radius, subject::generator_t* generator)
{
  math::ray_t* rays = (math::ray_t*)malloc(n_rays * sizeof(math::ray_t));
  if (rays == 0)
    throw std::out_of_range("not enough memory to allocate rays");
  face_t** hit_face = (face_t**)malloc(n_rays * sizeof(face_t*));
  math::vec3* hit_point = (math::vec3*)malloc(n_rays * sizeof(math::vec3));
  task_t* task = (task_t*)malloc(sizeof(task_t));
  *task = { n_rays, rays, hit_face, hit_point };
  for (int i = 0; i != n_rays; ++i)
  { 
    math::vec3 direction;
    generator_surface_point(generator, 1, &direction);
    direction *= (2 + radius);
    // Origin is scaled by 10% to direction - rays directed towards zero.
    math::vec3 origin = direction * 1.1f;
    rays[i] = { origin, direction };
  }

  return task;
}

/// @brief Clones given task. Uses to copy task from GPU to CPU.
task_t* task_clone(task_t* task)
{
  task_t* result = (task_t*)malloc(sizeof(task_t));
  result->n_tasks = task->n_tasks;
  result->ray = task->ray;
  result->hit_face = (face_t**)malloc(task->n_tasks * sizeof(face_t*));
  result->hit_point = (math::vec3*)malloc(task->n_tasks * sizeof(math::vec3));
  return result;
}

/// @brief Error struct for @see CalculateError(task_t* cpu, task_t* gpu).
struct ErrorStats
{
  ErrorStats()
  : TotalDistance(0)
  , MaxDistance(0)
  , AverageDistance(0)
  , Mismatch(0)
  {
  }

  float TotalDistance;
  float MaxDistance;
  float AverageDistance;
  int Mismatch;
};

/// @brief Checks errors between tasks in different ray caster implementations.
ErrorStats CalculateError(task_t* cpu, task_t* gpu)
{
  ErrorStats result;
  for (int i = 0; i != cpu->n_tasks; ++i)
  {
    if (cpu->hit_face[i] != gpu->hit_face[i])
    {
      ++result.Mismatch;
      continue;
    }

    if (cpu->hit_face[i] != 0)
    {
      math::vec3 diff = cpu->hit_point[i] - gpu->hit_point[i];
      float distance = sqrtf(dot(diff, diff));
      result.TotalDistance += distance;
      if (distance > result.MaxDistance)
        result.MaxDistance = distance;
    }
  }

  result.AverageDistance = result.TotalDistance / cpu->n_tasks;

  return result;
}

/// @brief Checks for non-equal ray intersections between tasks in different ray caster implementations.
int CalculateMismatch(task_t* cpu, task_t* gpu)
{
  int result = 0;
  for (int i = 0; i != cpu->n_tasks; ++i)
  {
    if (cpu->hit_face[i] != gpu->hit_face[i])
    {
      ++result;
    }
  }

  return result;
}

int main(int argc, char* argv[])
{
  try
  {
    int n_faces = 2000;
    int n_rays = 100 * n_faces;
    bool no_cpu = true;
    bool no_form_factors = true;
    bool no_radiance = false;

    float radius = 20;
    
    subject::generator_t* generator = subject::generator_create_cube();
    // Create systems for CPU and GPU.
    system_t* cuda_system = system_create(RAY_CASTER_SYSTEM_CUDA);
    system_t* cpu_system = system_create(RAY_CASTER_SYSTEM_CPU);
    emission::system_t* emitter = emission::system_create(EMISSION_MALLEY_CPU, cuda_system);
    printf("Generating confetti scene with %d elements...\n", n_faces);

    // Create random scene for ray caster.
    scene_t* scene = MakeConfettiScene(n_faces, radius, generator);
    printf("Generating %d collapsing rays...\n", n_rays);

    // Create task for ray caster.
    task_t* gpuTask = MakeCollapsingRays(n_rays, radius, generator);
    task_t* cpuTask = task_clone(gpuTask);

    // Create small warm-up scene for ray caster (from main confetti scene) and run ray caster on it.
    printf("Warming up...\n");
    scene_t warm_up_scene = *scene;
    warm_up_scene.n_faces = warm_up_scene.n_faces > 32 ? 32 : warm_up_scene.n_faces;
    task_t warm_up_task = *gpuTask;
    warm_up_task.n_tasks = 1;
    system_set_scene(cuda_system, &warm_up_scene);
    system_prepare(cuda_system);
    system_cast(cuda_system, &warm_up_task);

    // Prepare ray casters.
    system_set_scene(cuda_system, scene);
    system_prepare(cuda_system);
    system_set_scene(cpu_system, scene);
    system_prepare(cpu_system);

    StopWatchInterface *hTimer;
    sdkCreateTimer(&hTimer);
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    // Run GPU ray casting task.
    printf("Casting scene on GPU...\n");
    system_cast(cuda_system, gpuTask);
    sdkStopTimer(&hTimer);
    double gpuTime = 1.0e-3 * sdkGetTimerValue(&hTimer);
    printf("Done in %fs.\n", gpuTime);

    if (!no_cpu)
    {
      // Run CPU ray casting task.
      printf("Casting scene on CPU...\n");
      sdkResetTimer(&hTimer);
      sdkStartTimer(&hTimer);
      system_cast(cpu_system, cpuTask);
      sdkStopTimer(&hTimer);
      double cpuTime = 1.0e-3 * sdkGetTimerValue(&hTimer);
      printf("Done in %fs.\n", cpuTime);

      ErrorStats error = CalculateError(cpuTask, gpuTask);
      printf("Error is %f average, %f max, %f total.\n", error.AverageDistance, error.MaxDistance, error.TotalDistance);
      printf("Face hit mismatch count is %d.\n", error.Mismatch);
      printf("GPU/CPU performance ratio is %.3f.\n", cpuTime / gpuTime);
    }

    if (!no_form_factors)
    {
      // Run form factors calculation on Cuda's ray caster results.
      
      form_factors::system_t* calculator = form_factors::system_create(FORM_FACTORS_CPU, emitter);
      form_factors::scene_t ff_scene;
      ff_scene.n_faces = scene->n_faces;
      ff_scene.faces = scene->faces;

      int n_meshes = scene->n_faces;

      ff_scene.n_meshes = n_meshes;
      ff_scene.meshes = (form_factors::mesh_t*) malloc(n_meshes * sizeof(form_factors::mesh_t));
      for (int i = 0; i != n_meshes; ++i)
      {
        form_factors::mesh_t& mesh = ff_scene.meshes[i];
        mesh.first_idx = i;
        mesh.n_faces = 1;
      }

      printf("Calculating form factors on CPU...\n");
      sdkResetTimer(&hTimer);
      sdkStartTimer(&hTimer);

      // Finally calculate form factors.
      form_factors::system_set_scene(calculator, &ff_scene);
      form_factors::system_prepare(calculator);
      form_factors::task_t* task = form_factors::task_create(&ff_scene, n_rays);
      form_factors::system_calculate(calculator, task);

      sdkStopTimer(&hTimer);
      double cpuTime = 1.0e-3 * sdkGetTimerValue(&hTimer);
      printf("Done in %fs.\n", cpuTime);

      task_free(task);
      free(ff_scene.meshes);
    }

    if (!no_radiance)
    {
      const int rays_per_face = 100;
      radiance_equation::params_t equationParams;
      equationParams.emitter = emitter;
      equationParams.n_rays = rays_per_face * scene->n_faces;

      thermal_equation::system_t* equation = thermal_equation::system_create(THERMAL_EQUATION_RADIANCE_CPU, &equationParams);

      thermal_solution::params_t solutionParams = { 1, &equation };
      thermal_solution::system_t* solution = thermal_solution::system_create(THERMAL_SOLUTION_CPU_ADAMS, &solutionParams);

      subject::material_t materials[1];
      materials[0] = subject::black_body();

      subject::scene_t te_scene;
      te_scene.n_faces = scene->n_faces;
      te_scene.faces = scene->faces;

      int n_meshes = scene->n_faces;
      te_scene.n_meshes = n_meshes;
      te_scene.meshes = (subject::mesh_t*)malloc(n_meshes * sizeof(subject::mesh_t));
      float* temperatures = (float*)malloc(n_meshes * sizeof(float));

      for (int m = 0; m != n_meshes; ++m)
      {
        subject::mesh_t& mesh = te_scene.meshes[m];
        mesh.first_idx = m;
        mesh.n_faces = 1;
        mesh.material_idx = 0;

        temperatures[m] = 300;
      }

      te_scene.n_materials = 1;
      te_scene.materials = materials;

      thermal_solution::task_t* task = thermal_solution::task_create(te_scene.n_meshes);
      task->time_delta = 0.1f;

      FILE* scene_dump = fopen("perf_test_scene.obj", "w");
      obj_export::scene(scene_dump, &te_scene);
      fclose(scene_dump);

      int steps_count = 100;
      printf("Calculating %d steps of thermal solution using %d rays per face in average...\n", steps_count, rays_per_face);
      sdkResetTimer(&hTimer);
      sdkStartTimer(&hTimer);

      int r = 0;
      if ((r = system_set_scene(solution, &te_scene, temperatures)) < 0)
      {
        printf("Failed to calculate thermal solution step.");
        return r;
      }
      
      for (int step = 1; step < steps_count; ++step)
      {
        if ((r = system_calculate(solution, task)) < 0)
        {
          printf("Failed to calculate thermal solution step.");
          return r;
        }
      }
      

      sdkStopTimer(&hTimer);
      double cpuTime = 1.0e-3 * sdkGetTimerValue(&hTimer);
      printf("Done in %fs.\n", cpuTime);
      int solution_ray_casts = scene->n_faces * rays_per_face * steps_count;
      printf("Solution/Ray one ray casting times ratio is %.3f\n", (cpuTime / solution_ray_casts) / (gpuTime / n_rays));

      free(temperatures);
      free(te_scene.meshes);
      thermal_solution::task_free(task);
      thermal_solution::system_free(solution);
      thermal_equation::system_free(equation);
    }

    emission::system_free(emitter);
    system_free(cuda_system);
    system_free(cpu_system);

    scene_free(scene);
    free(gpuTask->ray);
    free(gpuTask->hit_face);
    free(gpuTask->hit_point);
    free(gpuTask);
    free(cpuTask->hit_face);
    free(cpuTask->hit_point);
    free(cpuTask);

    generator_free(generator);

    return 0;
  }
  catch (const std::exception& e)
  {
    printf("%s\n", e.what());
    return -1;
  }
}

