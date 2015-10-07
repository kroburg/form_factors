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

#include "gtest/gtest.h"

#include "math/operations.h"
#include "math/triangle.h"
#include "emission/system.h"
#include "form_factors/system.h"
#include "thermal_solution/system.h"
#include "thermal_equation/system.h"
#include "thermal_equation/conductive_cpu.h"
#include "thermal_equation/heat_source_cpu.h"
#include "subject/objects.h"

using namespace testing;

class ConductiveSourceSink
  : public Test
{
public:
  ConductiveSourceSink()
  {
    conductive_equation::params_t conductivityParams;
    Equations[0] = thermal_equation::system_create(THERMAL_EQUATION_CONDUCTIVE_CPU, &conductivityParams);

    Sources[0].mesh_idx = 0;
    Sources[0].power = 25;

    Sources[1].mesh_idx = 99;
    Sources[1].power = -25;

    heat_source_equation::params_t heatingParams = { 2, Sources };
    Equations[1] = thermal_equation::system_create(THERMAL_EQUATION_HEAT_SOURCE_CPU, &heatingParams);

    SolutionParams = { 2, Equations };
    System = thermal_solution::system_create(THERMAL_SOLUTION_CPU_ADAMS, &SolutionParams);

    Materials[0] = subject::material_Al(0.02f);
    Materials->shell.density = 8960;
    Materials->shell.heat_capacity = 386;
    Materials->shell.thermal_conductivity = 401;
    Materials->shell.thickness = 0.01f;

    Faces = subject::plane_grid_faces(1, 1, Dimension, Dimension);
    Meshes = subject::plane_grid_meshes(Dimension, Dimension);

    Scene = { Count * 2, Faces,
      Count, Meshes,
      1, Materials };

    for (int i = 0; i != Count; ++i)
      Temperatures[i] = 300;

    Task = thermal_solution::task_create(Scene.n_meshes);
    Task->time_delta = 1.f;
  }

  ~ConductiveSourceSink()
  { 
    thermal_solution::task_free(Task);
    thermal_solution::system_free(System);
    for (thermal_equation::system_t** e = Equations; e != Equations + 2; ++e)
      thermal_equation::system_free(*e);
    free(Meshes);
    free(Faces);
  }

  static const int Dimension = 10;
  static const int Count = Dimension * Dimension;

  heat_source_equation::heat_source_t Sources[2];
  thermal_equation::system_t* Equations[2];
  thermal_solution::params_t SolutionParams;
  thermal_solution::system_t* System;
  subject::material_t Materials[1];
  subject::face_t* Faces;
  subject::mesh_t* Meshes;
  subject::scene_t Scene;
  float Temperatures[Count];
  thermal_solution::task_t* Task;
};

TEST_F(ConductiveSourceSink, DISABLED_DistributeAsTheory)
{
  using namespace thermal_solution;

  Task->time_delta = .1f;

  int r = 0;
  r = system_set_scene(System, &Scene, Temperatures);
  float prevTemperature = Temperatures[0];
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  r = system_calculate(System, Task);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  

  for (int step = 1; step < 100000; ++step)
  {
   /* if (Task->temperatures[0] < 290 || Task->temperatures[0] > 310)
      __asm int 3;*/
    r = system_calculate(System, Task);
    ASSERT_EQ(THERMAL_SOLUTION_OK, r);
    float currentTemperature = Task->temperatures[0];
    if (currentTemperature / prevTemperature < 1.0000001f)
      break;
    prevTemperature = currentTemperature;
  }
  r = system_calculate(System, Task);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
}
