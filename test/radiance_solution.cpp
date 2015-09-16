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
#include "thermal_equation/radiance_cpu.h"
#include "thermal_equation/form_factors.h"
#include "subject/objects.h"

using namespace testing;

struct FormFactorsParameters
{
  FormFactorsParameters()
    : FormFactors(0)
    , Accuracy(0.01f)
  {
  }

  thermal_equation::system_t* CreateEquation(emission::system_t* emitter)
  {
    FormFactors = form_factors::system_create(FORM_FACTORS_CPU, emitter);
    EquationParams.form_factors_calculator = FormFactors;
    EquationParams.n_rays = 10000;
    return thermal_equation::system_create(THERMAL_EQUATION_SB_FF_CPU, &EquationParams);
  }

  ~FormFactorsParameters()
  {
    form_factors::system_free(FormFactors);
  }

  form_factors::system_t* FormFactors;
  form_factors_equation::params_t EquationParams;
  float Accuracy;
};

struct DirectMethodParameters
{
  DirectMethodParameters()
    : Accuracy(0.001f)
  {
  }

  thermal_equation::system_t* CreateEquation(emission::system_t* emitter)
  { 
    EquationParams.emitter = emitter;
    EquationParams.n_rays = 1000;
    return thermal_equation::system_create(THERMAL_EQUATION_RADIANCE_CPU, &EquationParams);
  }
  
  radiance_equation::params_t EquationParams;
  float Accuracy;
};

class RadianceSolution
  : public Test
  , public DirectMethodParameters
{
public:
  RadianceSolution()
  {
    RayCaster = ray_caster::system_create(RAY_CASTER_SYSTEM_CUDA);
    Emitter = emission::system_create(EMISSION_CPU, RayCaster);
    Equation = this->CreateEquation(Emitter);

    SolutionParams = { 1, &Equation };
    System = thermal_solution::system_create(THERMAL_SOLUTION_CPU_ADAMS, &SolutionParams);

    Materials[0] = subject::black_body();
    Materials[0].front.emissivity = 0;

    memcpy(Faces, subject::box(), sizeof(subject::face_t) * 12);
    memcpy(Faces + 12, subject::box(), sizeof(subject::face_t) * 12);

    Scale = 3;

    float factor = 1.f / Scale;
    float shift_value = 0.5f - factor / 2.f;
    math::vec3 shift = math::make_vec3(shift_value, shift_value, shift_value);
    for (int f = 0; f != 12; ++f)
    {
      subject::face_t& face = Faces[12 + f];
      face.points[0] *= factor;
      face.points[1] *= factor;
      face.points[2] *= factor;

      face.points[0] += shift;
      face.points[1] += shift;
      face.points[2] += shift;

      triangle_flip_normal(face);
    }

    Meshes[0] = { 0, 12, 0 };
    Meshes[1] = { 12, 12, 0 };
    Scene = { 24, Faces,
      2, Meshes,
      1, Materials };

    Temperatures[0] = 20.f;
    Temperatures[1] = 300.f;

    Task = thermal_solution::task_create(Scene.n_meshes);
    Task->time_delta = 0.1f;
  }

  ~RadianceSolution()
  {
    thermal_solution::task_free(Task);
    thermal_solution::system_free(System);
    thermal_equation::system_free(Equation);
    emission::system_free(Emitter);
    ray_caster::system_free(RayCaster);
  }
  
  ray_caster::system_t* RayCaster;
  emission::system_t* Emitter;
  thermal_equation::system_t* Equation;
  thermal_solution::params_t SolutionParams;
  thermal_solution::system_t* System;
  subject::material_t Materials[1];
  float Scale;
  subject::face_t Faces[24];
  subject::mesh_t Meshes[2];
  subject::scene_t Scene;
  float Temperatures[2];
  thermal_solution::task_t* Task;
};

TEST_F(RadianceSolution, BoxPreserveEnergy)
{
  using namespace thermal_solution;
  Scene.n_faces = 12;
  Scene.n_meshes = 1;

  const float initialEnergy = Temperatures[0];
  int r = 0;
  r = system_set_scene(System, &Scene, Temperatures);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  for (int step = 1; step < 50; ++step)
  {
    r = system_calculate(System, Task);
    ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  }
  r = system_calculate(System, Task);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  ASSERT_NEAR(1.f, Task->temperatures[0] / initialEnergy, this->Accuracy);
}

TEST_F(RadianceSolution, NestedBoxesPreserveEnergy)
{
  using namespace thermal_solution;
  const float initialEnergy = Temperatures[0] * Scale * Scale + Temperatures[1];

  int r = 0;
  r = system_set_scene(System, &Scene, Temperatures);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  for (int step = 1; step < 100; ++step)
  {
    r = system_calculate(System, Task);
    ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  }
  r = system_calculate(System, Task);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);

  float resultEnergy = Task->temperatures[0] * Scale * Scale + Task->temperatures[1];
  ASSERT_NEAR(1.f, resultEnergy / initialEnergy, this->Accuracy);
}
