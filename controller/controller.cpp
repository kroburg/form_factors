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
 * This module is a console utility that simple runs whole form factors
 * calculation with specified ray caster and calculator. Use it as an basic HOW-TO sample also.
 */

#include "../import_export/obj_import.h"
#include "../ray_caster/system.h"
#include "../thermal_solution/system.h"
#include "../thermal_equation/system.h"
#include "../thermal_equation/radiance_cpu.h"
#include "../import_export/obj_export.h"
#include <helper_timer.h>
#include <iostream>
#include <cstring>

#ifdef _WIN32
#include <tchar.h>
#endif

void PrintUsage()
{
  std::cout << "Usage: controller <input scene> <input task> <result output [-]> <rays_count [1000000]> <step count [100]> <ray_caster type:(cpu/cuda)[cpu]" << std::endl;
}

int main(int argc, char* argv[])
{
  if (argc < 3)
  {
    PrintUsage();
    return 1;
  }

  const char* input_scene = 0;
  const char* input_task = 0;
  const char* result_name = "-";
  int n_rays = 1000 * 1000;
  int n_steps = 100;
  int type = RAY_CASTER_SYSTEM_CPU;

  for (int i = 1; i < argc; ++i)
  {
    switch (i)
    {
    case 1:
      input_scene = argv[i];
      break;

    case 2:
      input_task = argv[i];
      break;

    case 3:
      result_name = argv[i];
      break;

    case 4:
      n_rays = atoi(argv[i]);
      break;

    case 5:
      n_steps = atoi(argv[i]);
      break;

    case 6:
      if (strcmp("cuda", argv[i]) == 0)
        type = RAY_CASTER_SYSTEM_CUDA;
      else if (strcmp("cpu", argv[i]) == 0)
        type = RAY_CASTER_SYSTEM_CPU;
      else
        return -RAY_CASTER_NOT_SUPPORTED;
    }
  }

  int r = 0;
  subject::scene_t* scene = 0;
  if ((r = obj_import::scene(input_scene, &scene)) != OBJ_IMPORT_OK)
  {
    std::cerr << "Failed to load scene " << input_scene << std::endl;
    return r;
  }

  thermal_solution::task_t* task = thermal_solution::task_create(0);
  task->time_delta = 0.1f;
  FILE* task_file = fopen(input_task, "r");
  if (!task_file)
  {
    fprintf(stderr, "Failed to load task '%s'\n", input_task);
    return -1;
  }
  else
  {
    r = obj_import::task(task_file, task);
    fclose(task_file);
    if (r < 0)
    {
      fprintf(stderr, "Failed to load task '%s'\n", input_task);
      return r;
    }
  }

  if (r < scene->n_meshes)
  {
    std::cerr << "There are not enough object temperatues defined in task. Initializing to default values." << std::endl;
    realloc(task->temperatures, scene->n_meshes * sizeof(float));
    for (; r != scene->n_meshes; ++r)
      task->temperatures[r] = 300;
  }

  ray_caster::system_t* caster = ray_caster::system_create(type);
  if (!caster)
  {
    std::cerr << "Failed to create ray caster" << std::endl;
    return 1;
  }

  emission::system_t* emitter = emission::system_create(EMISSION_CPU, caster);
  if (!emitter)
  {
    std::cerr << "Failed to create emitter" << std::endl;
    return 1;
  }

  radiance_equation::params_t equationParams;
  equationParams.emitter = emitter;
  equationParams.n_rays = n_rays;

  thermal_equation::system_t* equation = thermal_equation::system_create(THERMAL_EQUATION_RADIANCE_CPU, &equationParams);
  if (!equation)
  {
    std::cerr << "Failed to create radiance equation" << std::endl;
    return 1;
  }

  thermal_solution::params_t solutionParams = { 1, &equation };
  thermal_solution::system_t* solution = thermal_solution::system_create(THERMAL_SOLUTION_CPU_ADAMS, &solutionParams);
  if (!solution)
  {
    std::cerr << "Failed to create thermal solution" << std::endl;
    return 1;
  }

  FILE* result_file = result_name[0] == '-' ? stdout : fopen(result_name, "w");

  obj_export::task(result_file, scene->n_meshes, task);
  
  StopWatchInterface *hTimer;
  sdkCreateTimer(&hTimer);
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  if ((r = system_set_scene(solution, scene, task->temperatures)) < 0)
  {
    printf("Failed to set thermal solution scene.");
    return r;
  }

  for (int step = 0; step < n_steps; ++step)
  {
    if ((r = system_calculate(solution, task)) < 0)
    { 
      std::cerr << "Failed to calculate thermal solution step" << std::endl; 
      return 1;
    }

    obj_export::task(result_file, scene->n_meshes, task);
  }
  
  sdkStopTimer(&hTimer);
  double cpuTime = 1.0e-3 * sdkGetTimerValue(&hTimer);
  printf("#Done in %fs.\n", cpuTime);  

  if (result_name[0] != '-')
    fclose(result_file);

  thermal_solution::task_free(task);
  thermal_solution::system_free(solution);
  thermal_equation::system_free(equation);
  emission::system_free(emitter);
  ray_caster::system_free(caster);
  
	return 0;
}

