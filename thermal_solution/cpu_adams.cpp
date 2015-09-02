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
* This module contains thermal solution using Adams integration method implemented on CPU.
*/

#include "cpu_adams.h"
#include "../thermal_equation/system.h"
#include "../math/triangle.h"
#include <stdlib.h>
#include <cstring>

namespace cpu_adams
{
  /// @brief Extended base system_t (C-style polymorphism)
  struct cpu_system_t : thermal_solution::system_t
  {
    thermal_solution::params_t params;
    thermal_solution::scene_t* scene;

    int n_step;
    float* temperatures;
    float* energy;
  };

  /// @brief Initializes system with given ray caster after creation.
  int init(cpu_system_t* system, thermal_solution::params_t* params)
  {
    system->params = *params;
    system->scene = 0;

    system->n_step = 0;
    system->temperatures = 0;
    system->energy = 0;
    
    return THERMAL_SOLUTION_OK;
  }

  /// @brief Shutdowns calculator system prior to free memory.
  int shutdown(cpu_system_t* system)
  {    
    system->scene = 0;

    system->n_step = 0;

    free(system->temperatures);
    system->temperatures = 0;

    free(system->energy);
    system->energy = 0;

    return THERMAL_SOLUTION_OK;
  }

  int set_scene(cpu_system_t* system, thermal_solution::scene_t* scene, float* temperatures)
  {
    system->scene = scene;

    int r = 0;
    for (int i = 0; i != system->params.n_equations; ++i)
    {
      thermal_equation::system_t* equation = system->params.equations[i];
      if ((r = thermal_equation::system_set_scene(equation, scene)) < 0)
        return r;
    }

    system->n_step = 0;

    free(system->temperatures);
    system->temperatures = (float*)malloc(5 * sizeof(float) * scene->n_meshes);
    memset(system->temperatures, 0, 5 * sizeof(float) * scene->n_meshes);
    memcpy(system->temperatures, temperatures, sizeof(float) * scene->n_meshes);

    free(system->energy);
    system->energy = (float*)malloc(5 * sizeof(float) * scene->n_meshes);
    memset(system->energy, 0, 5 * sizeof(float) * scene->n_meshes);

    return THERMAL_SOLUTION_OK;
  }

  float* get_step_energy(cpu_system_t* system, int n_step)
  {
    int row = n_step % 5;
    return &system->energy[row * system->scene->n_meshes];
  }

  float* get_step_temperatures(cpu_system_t* system, int n_step)
  {
    int row = n_step % 5;
    return &system->temperatures[row * system->scene->n_meshes];
  }

  float mesh_area(cpu_system_t* system, int mesh_idx)
  {
    float area = 0;
    const thermal_solution::mesh_t& mesh = system->scene->meshes[mesh_idx];
    for (int f = 0; f != mesh.n_faces; ++f)
    {
      area += math::triangle_area(system->scene->faces[mesh.first_idx + f]);
    }
    return area;
  }

  int calculate_energy(cpu_system_t* system, float* temperatures, float* energy)
  {
    thermal_equation::task_t* task = thermal_equation::task_create(system->scene);
    task->temperatures = temperatures;
    int r = 0;
    for (int i = 0; i != system->params.n_equations; ++i)
    {
      thermal_equation::system_t* equation = system->params.equations[i];
      if ((r = thermal_equation::system_calculate(equation, task)) < 0)
      {
        thermal_equation::task_free(task);
        return r;
      }
    }

    const int n_meshes = system->scene->n_meshes;
    for (int m = 0; m != n_meshes; ++m)
    {
      float power_balance = task->absorption[m] - task->emission[m];
      const thermal_solution::material_t* material = &system->scene->materials[system->scene->meshes[m].material_idx];
      /// @todo Precalculate areas and store with mesh (required for form_factors, sb_ff_te and here).
      const float C = mesh_area(system, m) * material->thickness * material->density * material->c;
      energy[m] = power_balance / C;
    }

    thermal_equation::task_free(task);
    return THERMAL_SOLUTION_OK;
  }

  const float u_matrix[5][5] = {
    {1.f,        0,           0,       0,           0},
    {3.f/2,      -1.f/2,      0,       0,           0},
    {23.f/12,    -4.f/3,      5.f/12,   0,          0},
    {55.f/24,    -59.f/24,    37.f/24,  -3.f/8,     0},
    {1901.f/720, -1387.f/360, 109.f/30, -637.f/360, 251.f/720}
  };

  /**
  *  @brief Calculates thermal solution step for a given scene.
  */
  int calculate(cpu_system_t* system, thermal_solution::task_t* task)
  {    
    int n = system->n_step;
    int k = 4;
    if (n < k)
      k = n;
    const float* u = u_matrix[k];

    const int n_meshes = system->scene->n_meshes;

    float* Tresult = task->temperatures;
    float* En = get_step_energy(system, n);
    float* Tn = get_step_temperatures(system, n);
    
    int r = 0;
    if ((r = calculate_energy(system, Tn, En)) < 0)
      return r;

    for (int m = 0; m != n_meshes; ++m)
    {
      float sum = 0;
      for (int l = 0; l <= k; ++l)
      {
        float* E = get_step_energy(system, n - l);
        sum += u[l] * E[m];
      }

      Tresult[m] = Tn[m] + task->time_delta * sum;
    }

    float* Tnext = get_step_temperatures(system, n + 1);
    memcpy(Tnext, Tresult, sizeof(float) * n_meshes);
    task->n_step = ++system->n_step;
 
    return THERMAL_SOLUTION_OK;
  }

  /// @brief Creates virtual methods table from local methods.
  const thermal_solution::system_methods_t methods =
  {
    (int(*)(thermal_solution::system_t* system, thermal_solution::params_t* params))&init,
    (int(*)(thermal_solution::system_t* system))&shutdown,
    (int(*)(thermal_solution::system_t* system, thermal_solution::scene_t* scene, float* temperatures))&set_scene,
    (int(*)(thermal_solution::system_t* system, thermal_solution::task_t* task))&calculate,
  };

  thermal_solution::system_t* system_create()
  {
    cpu_system_t* s = (cpu_system_t*)malloc(sizeof(cpu_system_t));
    s->methods = &methods;
    return s;
  }
}
