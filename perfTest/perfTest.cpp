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

#include <stdio.h>
#include <random>
#include <cmath>

#ifdef _WIN32
#include <tchar.h>
#endif

#include "../ray_caster/system.h"
#include "../form_factors/system.h"
#include "../math/operations.h"

#include <helper_timer.h>

using namespace ray_caster;

class ShperePointsGenerator
{
public:
  ShperePointsGenerator()
    : ThetaGenerator(0)
    , UGenerator(1)
    , RGenerator(2)
    , ThetaDistribution(0, float(M_PI * 2.))
    , UDistribution(-1, 1)
    , RDistribution(-100, 100)
  {
  }

  math::vec3 Point()
  {
    // http://mathworld.wolfram.com/SpherePointPicking.html

    float theta = ThetaDistribution(ThetaGenerator);
    float u = UDistribution(UGenerator);

    float c = sqrtf(1 - u * u);

    float x = c * cos(theta);
    float y = c * sin(theta);
    float z = u;

    return math::make_vec3(x, y, z);
  }

  math::vec3 Position()
  {
    return Point() * RDistribution(RGenerator);
  }

private:
  std::mt19937 ThetaGenerator;
  std::mt19937 UGenerator;
  std::mt19937 RGenerator;
  std::uniform_real_distribution<float> ThetaDistribution;
  std::uniform_real_distribution<float> UDistribution;
  std::uniform_real_distribution<float> RDistribution;
};

scene_t* MakeConfettiScene(int n_faces, ShperePointsGenerator& generator)
{
  face_t* faces = (face_t*)malloc(n_faces * sizeof(face_t));
  scene_t* scene = (scene_t*)malloc(sizeof(scene_t));
  *scene = { n_faces, faces };
  for (int i = 0; i != n_faces; ++i)
  {
    face_t& f = faces[i];
    f.points[0] = generator.Point();
    f.points[1] = generator.Point();
    f.points[2] = generator.Point();
    math::vec3 positon = generator.Position();
    f.points[0] += positon;
    f.points[1] += positon;
    f.points[2] += positon;
  }  
  
  return scene;
}

task_t* MakeCollapsingRays(int n_rays, ShperePointsGenerator& generator)
{
  math::ray_t* rays = (math::ray_t*)malloc(n_rays * sizeof(math::ray_t));
  face_t** hit_face = (face_t**)malloc(n_rays * sizeof(face_t*));
  math::vec3* hit_point = (math::vec3*)malloc(n_rays * sizeof(math::vec3));
  task_t* task = (task_t*)malloc(sizeof(task_t));
  *task = { n_rays, rays, hit_face, hit_point };
  for (int i = 0; i != n_rays; ++i)
  { 
    math::vec3 direction = generator.Point() * 110.f;
    math::vec3 origin = direction * 1.1f;
    
    rays[i] = { origin, direction };
  }

  return task;
}

task_t* task_clone(task_t* task)
{
  task_t* result = (task_t*)malloc(sizeof(task_t));
  result->n_tasks = task->n_tasks;
  result->ray = task->ray;
  result->hit_face = (face_t**)malloc(task->n_tasks * sizeof(face_t*));
  result->hit_point = (math::vec3*)malloc(task->n_tasks * sizeof(math::vec3));
  return result;
}

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
  int n_faces = 1000;
  int n_rays = 10000 * n_faces;
  bool no_cpu = true;
  bool no_form_factors = false;

  ShperePointsGenerator generator;
  system_t* cuda_system = system_create(RAY_CASTER_SYSTEM_CUDA);
  system_t* cpu_system = system_create(RAY_CASTER_SYSTEM_CPU);
  printf("Generating confetti scene with %d elements...\n", n_faces);
  scene_t* scene = MakeConfettiScene(n_faces, generator);
  printf("Generating %d collapsing rays...\n", n_rays);
  task_t* gpuTask = MakeCollapsingRays(n_rays, generator);
  task_t* cpuTask = task_clone(gpuTask);

  printf("Warming up...\n");
  scene_t warm_up_scene = *scene;
  warm_up_scene.n_faces = warm_up_scene.n_faces > 32 ? 32 : warm_up_scene.n_faces;
  task_t warm_up_task = *gpuTask;
  warm_up_task.n_tasks = 1;
  system_set_scene(cuda_system, &warm_up_scene);
  system_prepare(cuda_system);
  system_cast(cuda_system, &warm_up_task);

  system_set_scene(cuda_system, scene);
  system_prepare(cuda_system);
  system_set_scene(cpu_system, scene);
  system_prepare(cpu_system);

  StopWatchInterface *hTimer;
  sdkCreateTimer(&hTimer);
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  printf("Casting scene on GPU...\n");
  system_cast(cuda_system, gpuTask);
  sdkStopTimer(&hTimer);
  double gpuTime = 1.0e-3 * sdkGetTimerValue(&hTimer);
  printf("Done in %fs.\n", gpuTime);

  if (!no_cpu)
  {
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
    printf("GPU/CPU performance ratio is %f.\n", cpuTime / gpuTime);
  }

  if (!no_form_factors)
  {
    form_factors::system_t* calculator = form_factors::system_create(FORM_FACTORS_CPU, cuda_system);
    form_factors::scene_t calculator_scene;
    calculator_scene.n_faces = scene->n_faces;
    calculator_scene.faces = scene->faces;

    int n_meshes = scene->n_faces;
    calculator_scene.n_meshes = n_meshes;
    calculator_scene.meshes = (form_factors::mesh_t*) malloc(n_meshes * sizeof(form_factors::mesh_t));
    for (int i = 0; i != n_meshes; ++i)
    {
      form_factors::mesh_t& mesh = calculator_scene.meshes[i];
      mesh.first_idx = i;
      mesh.n_faces = 1;
    }

    printf("Calculating form factors on CPU...\n");
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    form_factors::system_set_scene(calculator, &calculator_scene);
    form_factors::system_prepare(calculator);
    form_factors::task_t* task = form_factors::task_create(&calculator_scene, n_rays);
    form_factors::system_calculate(calculator, task);

    sdkStopTimer(&hTimer);
    double cpuTime = 1.0e-3 * sdkGetTimerValue(&hTimer);
    printf("Done in %fs.\n", cpuTime);

    task_free(task);
    free(calculator_scene.meshes);
  }
  
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

	return 0;
}


#ifdef _WIN32
int _tmain(int argc, _TCHAR* argv[])
{
  return main(argc, argv);
}
#endif
