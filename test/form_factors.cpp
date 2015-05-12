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

#include "ray_caster/system.h"
#include "form_factors/system.h"
#include "math/operations.h"
#include "math/triangle.h"

using math::vec3;
using math::make_vec3;
using namespace testing;
using namespace form_factors;

template <int EngineTypeID>
struct EngineType
{
  static const int ID = EngineTypeID;
};

template <typename EngineType>
class FormFactors : public Test
{
public:
  FormFactors()
  {
    Scene = 0;
    RayCaster = ray_caster::system_create(RAY_CASTER_SYSTEM_CPU);
    Calculator = system_create(EngineType::ID, RayCaster);
  }

  ~FormFactors()
  {
    system_free(Calculator);
    ray_caster::system_free(RayCaster);
    scene_free(Scene);
  }

  face_t make_floor_face1(math::vec3 offset)
  {
    vec3 a = { 0.f, 0.f, 0.f };
    vec3 b = { 1.f, 0.f, 0.f };
    vec3 c = { 0.f, 1.f, 0.f };

    return make_face(a + offset, b + offset, c + offset);
  }

  face_t make_floor_face2(math::vec3 offset)
  {
    vec3 a = { 1.f, 0.f, 0.f };
    vec3 b = { 1.f, 1.f, 0.f };
    vec3 c = { 0.f, 1.f, 0.f };

    return make_face(a + offset, b + offset, c + offset);
  }

  scene_t* MakeParallelPlanesScene()
  {
    int n_faces = 4;
    int n_meshes = 2;
    float c = 1;
    face_t* faces = (face_t*)malloc(n_faces * sizeof(face_t));
    faces[0] = make_floor_face1(math::make_vec3(0, 0, 0));
    faces[1] = make_floor_face2(math::make_vec3(0, 0, 0));
    faces[2] = make_floor_face1(math::make_vec3(0, 0, c));
    faces[3] = make_floor_face2(math::make_vec3(0, 0, c));

    mesh_t* meshes = (mesh_t*)malloc(n_meshes * sizeof(mesh_t));
    meshes[0] = { 0, 2 };
    meshes[1] = { 2, 2 };

    scene_t* s = scene_create();
    Scene = s;
    *s = { n_faces, faces, n_meshes, meshes };
    return s;
  }

  scene_t* Scene;
  ray_caster::system_t* RayCaster;
  system_t* Calculator;
};

typedef ::testing::Types<EngineType<FORM_FACTORS_CPU> > FormFactorsTypes;
TYPED_TEST_CASE(FormFactors, FormFactorsTypes);

math::point_t theor_parallel_planes(math::point_t a, math::point_t b, math::point_t c) {
  auto x = a / c, y = b / c, xq = x * x, yq = y * y;
  auto result = 2 / math::point_t(M_PI) / x / y * (logf(sqrtf((1 + xq) * (1 + yq) / (1 + xq + yq))) + x * sqrtf(1 + yq) * atanf(x / sqrtf(1 + yq)) + y * sqrtf(1 + xq) * atanf(y / sqrtf(1 + xq)) - x * atanf(x) - y * atanf(y));
  return result / 2;
}

TYPED_TEST(FormFactors, ParallelPlanesCorrect)
{
  scene_t* stackScene = MakeParallelPlanesScene();

  system_set_scene(Calculator, stackScene);
  ASSERT_EQ(FORM_FACTORS_OK, system_prepare(Calculator));

  float factors[2 * 2];
  task_t task = { 40000, factors };

  float theoretical = theor_parallel_planes(1, 1, 1);

  ASSERT_EQ(FORM_FACTORS_OK, system_calculate(Calculator, &task));
  EXPECT_NEAR(theoretical, factors[1], 0.01);
  EXPECT_NEAR(theoretical, factors[2], 0.01);
}
