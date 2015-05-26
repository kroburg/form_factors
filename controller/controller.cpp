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

#include "../import_export/obj_import.h"
#include "../ray_caster/system.h"
#include "../form_factors/system.h"
#include "../import_export/csv_export.h"
#include <helper_timer.h>
#include <iostream>
#include <cstring>

#ifdef _WIN32
#include <tchar.h>
#endif

void PrintUsage()
{
  std::cout << "Usage: controller <input obj> <output csv> <rays_count [1000000]> <ray_caster type:(cpu/cuda)[cpu]" << std::endl;
}

int main(int argc, char* argv[])
{
  if (argc < 3)
  {
    PrintUsage();
    return 1;
  }

  const char* input = 0;
  const char* output = 0;
  int n_rays = 1000 * 1000;
  int type = RAY_CASTER_SYSTEM_CPU;

  for (int i = 1; i < argc; ++i)
  {
    switch (i)
    {
    case 1:
      input = argv[i];
      break;

    case 2:
      output = argv[i];
      break;

    case 3:
      n_rays = atoi(argv[i]);
      break;

    case 4:
      if (strcmp("cuda", argv[i]) == 0)
        type = RAY_CASTER_SYSTEM_CUDA;
      else if (strcmp("cpu", argv[i]) == 0)
        type = RAY_CASTER_SYSTEM_CPU;
      else
        return -RAY_CASTER_NOT_SUPPORTED;
    }
  }

  int r = 0;
  ray_caster::system_t* caster = ray_caster::system_create(type);
  if (!caster)
  {
    std::cerr << "Failed to create ray caster" << std::endl;
    return 1;
  }

  form_factors::system_t* calculator = form_factors::system_create(FORM_FACTORS_CPU, caster);
  if (!calculator)
  {
    std::cerr << "Failed to create form factors calculator" << std::endl;
    return 1;
  }

  form_factors::scene_t* scene = 0;
  if ((r = obj_import::import_obj(input, &scene)) != OBJ_IMPORT_OK)
  {
    std::cerr << "Failed to load scene " << input << std::endl;
    return r;
  }

  form_factors::task_t* task = form_factors::task_create(scene, n_rays);
  if (!task)
  {
    std::cerr << "Failed to create calculator task" << std::endl;
    return 1;
  }

  if ((r = form_factors::system_set_scene(calculator, scene)) != FORM_FACTORS_OK)
  {
    std::cerr << "Failed to set scene." << std::endl;
    return r;
  }

  if ((r = form_factors::system_prepare(calculator)) != FORM_FACTORS_OK)
  {
    std::cerr << "Failed to prepare scene." << std::endl;
    return r;
  }

  StopWatchInterface *hTimer;
  sdkCreateTimer(&hTimer);
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  printf("Calculating form factors...\n");
  if ((r = form_factors::system_calculate(calculator, task)) != FORM_FACTORS_OK)
  {
    std::cerr << "Failed to calculate form factors." << std::endl;
    return r;
  }
  sdkStopTimer(&hTimer);
  double cpuTime = 1.0e-3 * sdkGetTimerValue(&hTimer);
  printf("Done in %fs.\n", cpuTime);

  if ((r = csv_export::export_csv(output, scene, task)) != CSV_EXPORT_OK)
  {
    std::cerr << "Failed to export csv." << std::endl;
    return r;
  }

  form_factors::task_free(task);
  form_factors::scene_free(scene);
  ray_caster::system_free(caster);
  form_factors::system_free(calculator);
  
	return 0;
}

