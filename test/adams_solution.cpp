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

#include "thermal_solution/system.h"
#include "thermal_equation/system.h"

using namespace testing;

namespace test_equation
{
  struct params_t {};

  /// @brief Extended base system_t (C-style polymorphism)
  struct cpu_system_t : thermal_equation::system_t
  {
    thermal_equation::scene_t* scene;
  };

  /// @brief Initializes system with given ray caster after creation.
  int init(cpu_system_t* system, params_t* params)
  {
    system->scene = 0;

    return THERMAL_EQUATION_OK;
  }

  /// @brief Shutdowns calculator system prior to free memory.
  int shutdown(cpu_system_t* system)
  {
    system->scene = 0;
    return THERMAL_EQUATION_OK;
  }

  int set_scene(cpu_system_t* system, thermal_equation::scene_t* scene)
  {
    system->scene = scene;
    return THERMAL_EQUATION_OK;
  }

  int calculate(cpu_system_t* system, thermal_equation::task_t* task)
  {
    //@todo Note n^2 complexity.
    const int n_meshes = system->scene->n_meshes;
    for (int m = 0; m != n_meshes; ++m)
    {
      task->emission[m] += 2.f;
      task->absorption[m] += 1.f;
    }

    return THERMAL_EQUATION_OK;
  }

  /// @brief Creates virtual methods table from local methods.
  const thermal_equation::system_methods_t methods =
  {
    (int(*)(thermal_equation::system_t* system, void* params))&init,
    (int(*)(thermal_equation::system_t* system))&shutdown,
    (int(*)(thermal_equation::system_t* system, thermal_equation::scene_t* scene))&set_scene,
    (int(*)(thermal_equation::system_t* system, thermal_equation::task_t* task))&calculate,
  };

  thermal_equation::system_t* system_create()
  {
    cpu_system_t* s = (cpu_system_t*)malloc(sizeof(cpu_system_t));
    s->methods = &methods;
    return s;
  }
}

class AdamsSolution : public Test
{
public:
  AdamsSolution()
  {
    Equation = test_equation::system_create();
    Params = { 1, &Equation };
    // @todo Remove params (and init()) from system_create()?
    System = thermal_solution::system_create(THERMAL_SOLUTION_CPU_ADAMS, &Params);

    Materials[0] = { 1.f, 1.f, 1.f };
    Faces[0] = MakeFloorFace1();
    Meshes[0] = { 0, 1, 0 };
    Scene = { 1, Faces,
      1, Meshes,
      1, Materials };

    Temperatures[0] = 300.f;

    Task = thermal_solution::task_create(&Scene);
  }

  ~AdamsSolution()
  {
    thermal_solution::task_free(Task);
    thermal_solution::system_free(System);
    thermal_equation::system_free(Equation);
  }

  /// @Face of area ~1.
  thermal_solution::face_t MakeFloorFace1()
  {
    using namespace math;
    vec3 a = { 0.f, 0.f, 0.f };
    vec3 b = { 1.41421f, 0.f, 0.f };
    vec3 c = { 0.f, 1.41421f, 0.f };

    return make_face(a, b, c);
  }

  thermal_equation::system_t* Equation;

  thermal_solution::params_t Params;
  thermal_solution::system_t* System;
  thermal_solution::material_t Materials[1];
  thermal_solution::face_t Faces[1];
  thermal_solution::mesh_t Meshes[1];
  thermal_solution::scene_t Scene;
  float Temperatures[1];
  thermal_solution::task_t* Task;
  
};

TEST(BasicAdamsSolution, ImplementCreateShutdown)
{
  thermal_solution::params_t params = { 0, 0 };
  thermal_solution::system_t* system = thermal_solution::system_create(THERMAL_SOLUTION_CPU_ADAMS, &params);
  thermal_solution::system_free(system);
}

TEST(BasicAdamsSolution, AcceptSimpleEquation)
{
  thermal_equation::system_t* equation = test_equation::system_create();
  thermal_solution::params_t params = { 1, &equation };
  thermal_solution::system_t* system = thermal_solution::system_create(THERMAL_SOLUTION_CPU_ADAMS, &params);
  thermal_solution::system_free(system);
  thermal_equation::system_free(equation);
}

TEST_F(AdamsSolution, AcceptSimpleScene)
{
  using namespace thermal_solution;
  system_set_scene(System, &Scene, Temperatures);
}

TEST_F(AdamsSolution, ImplementFirstIntegrationStep)
{
  using namespace thermal_solution;
  int r = 0;
  r = system_set_scene(System, &Scene, Temperatures);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  r = system_calculate(System, Task);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  ASSERT_EQ(1, Task->n_step);
  EXPECT_NEAR(299.f, Task->temperatures[0], 0.01f);

}

TEST_F(AdamsSolution, ImplementFirstFiveIntegrationSteps)
{
  using namespace thermal_solution;
  int r = 0;
  r = system_set_scene(System, &Scene, Temperatures);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);

  for (int step = 1; step < 6; ++step)
  {
    r = system_calculate(System, Task);
    ASSERT_EQ(THERMAL_SOLUTION_OK, r);
    ASSERT_EQ(step, Task->n_step);
    EXPECT_NEAR(300.f - step, Task->temperatures[0], 0.01f);
  }
}

TEST_F(AdamsSolution, ImplementRingbufferReuse)
{
  using namespace thermal_solution;
  int r = 0;
  r = system_set_scene(System, &Scene, Temperatures);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);

  for (int step = 1; step < 12; ++step)
  {
    r = system_calculate(System, Task);
    ASSERT_EQ(THERMAL_SOLUTION_OK, r);
    ASSERT_EQ(step, Task->n_step);
    EXPECT_NEAR(300.f - step, Task->temperatures[0], 0.01f);
  }
}