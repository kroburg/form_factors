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
#include "math/operations.h"
#include "math/triangle.h"

using namespace ray_caster;
using math::vec3;
using math::make_vec3;
using namespace testing;

template <int EngineTypeID>
struct EngineType
{
  static const int ID = EngineTypeID;
};

template <typename EngineType>
class RayCaster : public Test
{
public:
  RayCaster()
  {
    Scene = 0;
    System = system_create(EngineType::ID);
  }

  ~RayCaster()
  {
    system_free(this->System);
    scene_free(Scene);
  }

  scene_t* MakeEmptyScene()
  {
    return scene_create();
  }

  face_t make_floor_face1()
  {
    vec3 a = { 0.f, 0.f, 0.f };
    vec3 b = { 1.f, 0.f, 0.f };
    vec3 c = { 0.f, 1.f, 0.f };

    return make_face(a, b, c);
  }

  face_t make_floor_face2()
  { 
    vec3 a = { 1.f, 0.f, 0.f };
    vec3 b = { 0.f, 1.f, 0.f };
    vec3 c = { 1.f, 1.f, 0.f };

    return make_face(a, b, c);
  }

  face_t make_stack_face()
  {
    vec3 a = { 0.f, 0.f, -1.f };
    vec3 b = { 1.f, 0.f, -1.f };
    vec3 c = { 0.f, 1.f, -1.f };

    return make_face(a, b, c);
  }

  scene_t* MakeSceneFromFaces(int n_faces, face_t* faces)
  {
    scene_t* s = scene_create();
    *s = { /*.n_faces =*/ n_faces, /*.faces =*/ faces };
    Scene = s;
    return s;
  }

  scene_t* MakeFloorScene()
  {
    int n_faces = 2;
    face_t* faces = (face_t*)malloc(n_faces * sizeof(face_t));
    faces[0] = make_floor_face1();
    faces[1] = make_floor_face2();

    return MakeSceneFromFaces(n_faces, faces);
  }

  scene_t* MakeStackScene()
  {
    int n_faces = 2;
    face_t* faces = (face_t*)malloc(n_faces * sizeof(face_t));
    faces[0] = make_floor_face1();
    faces[1] = make_stack_face();

    return MakeSceneFromFaces(n_faces, faces);
  }

  scene_t* Scene;
  system_t* System;
};

typedef ::testing::Types<EngineType<RAY_CASTER_SYSTEM_CPU>, EngineType<RAY_CASTER_SYSTEM_CUDA> > RayCasterTypes;
TYPED_TEST_CASE(RayCaster, RayCasterTypes);

TYPED_TEST(RayCaster, MemoryManagementIsCorrect)
{
  // @todo how to check?
  // Not crashing is ok already.
}

TYPED_TEST(RayCaster, AcceptEmptyScene)
{
  scene_t* emptyScence = this->MakeEmptyScene();
  ASSERT_EQ(RAY_CASTER_OK, system_set_scene(this->System, emptyScence));
}

TYPED_TEST(RayCaster, AcceptFloorScene)
{
  scene_t* floorScene = MakeFloorScene();
  ASSERT_EQ(RAY_CASTER_OK, system_set_scene(this->System, floorScene));
}

TYPED_TEST(RayCaster, PrepareFailsForNoScene)
{ 
  ASSERT_EQ(-RAY_CASTER_ERROR, system_prepare(this->System));
}

TYPED_TEST(RayCaster, PrepareFailsForEmptyScene)
{
  scene_t* emptyScence = this->MakeEmptyScene();
  system_set_scene(this->System, emptyScence);
  ASSERT_EQ(-RAY_CASTER_ERROR, system_prepare(this->System));
}

TYPED_TEST(RayCaster, PreparePassForNotEmptyScene)
{
  scene_t* floorScene = MakeFloorScene();
  system_set_scene(this->System, floorScene);
  ASSERT_EQ(RAY_CASTER_OK, system_prepare(this->System));
}

TYPED_TEST(RayCaster, ProcessAllRays)
{
  scene_t* floorScene = MakeFloorScene();

  system_set_scene(this->System, floorScene);
  ASSERT_EQ(RAY_CASTER_OK, system_prepare(this->System));

  ray_t ray[2];
  face_t* hit_face[2];
  vec3 hit_point[2];

  vec3 center0 = triangle_center(floorScene->faces[0]);
  vec3 origin0 = center0 + make_vec3(0.f, 0.f, 1.f);
  ray[0] = ray_to_triangle(origin0, floorScene->faces[0]);

  vec3 center1 = triangle_center(floorScene->faces[1]);
  vec3 origin1 = center1 + make_vec3(0.f, 0.f, -1.f);
  ray[1] = ray_to_triangle(origin1, floorScene->faces[1]);

  task_t task = { 2, ray, hit_face, hit_point };

  ASSERT_EQ(RAY_CASTER_OK, system_cast(this->System, &task));

  ASSERT_TRUE(near_enough(hit_point[0], center0));
  ASSERT_TRUE(near_enough(hit_point[1], center1));
}

TYPED_TEST(RayCaster, JustWorks)
{
  scene_t* floorScene = MakeFloorScene();

  system_set_scene(this->System, floorScene);
  ASSERT_EQ(RAY_CASTER_OK, system_prepare(this->System));

  vec3 center = triangle_center(floorScene->faces[0]);
  vec3 origin = center + make_vec3(0.f, 0.f, 1.f);
  ray_t ray = ray_to_triangle(origin, floorScene->faces[0]);
  face_t* hit_face = 0;
  vec3 hit_point;
  task_t task = { 1, &ray, &hit_face, &hit_point };
  
  ASSERT_EQ(RAY_CASTER_OK, system_cast(this->System, &task));
  // @todo Provide gtest comparison overloads for vec3
  ASSERT_TRUE(near_enough(hit_point, center));
}

TYPED_TEST(RayCaster, HandleRayIntersectingTriangle)
{
  scene_t* stackScene = MakeStackScene();

  system_set_scene(this->System, stackScene);
  ASSERT_EQ(RAY_CASTER_OK, system_prepare(this->System));

  vec3 center = triangle_center(stackScene->faces[0]);
  vec3 origin = center + make_vec3(0.f, 0.f, .5f);
  vec3 direction = center - make_vec3(0.f, 0.f, .5f);
  ray_t ray = { origin, direction };
  face_t* hit_face = 0;
  vec3 hit_point;
  task_t task = { 1, &ray, &hit_face, &hit_point };

  ASSERT_EQ(RAY_CASTER_OK, system_cast(this->System, &task));
  // @todo Provide gtest comparison overloads for vec3
  ASSERT_TRUE(near_enough(hit_point, center));
}

TYPED_TEST(RayCaster, FindNearestTriangle)
{
  scene_t* stackScene = MakeStackScene();

  system_set_scene(this->System, stackScene);
  ASSERT_EQ(RAY_CASTER_OK, system_prepare(this->System));

  vec3 center = triangle_center(stackScene->faces[0]);
  vec3 origin = center + make_vec3(0.f, 0.f, 1.f);
  ray_t ray = ray_to_triangle(origin, stackScene->faces[0]);
  face_t* hit_face = 0;
  vec3 hit_point;
  task_t task = { 1, &ray, &hit_face, &hit_point };

  ASSERT_EQ(RAY_CASTER_OK, system_cast(this->System, &task));
  // @todo Provide gtest comparison overloads for vec3
  ASSERT_TRUE(near_enough(hit_point, center));
}

TYPED_TEST(RayCaster, SkipTriangleNearButInOppositeDirection)
{
  scene_t* stackScene = MakeStackScene();

  system_set_scene(this->System, stackScene);
  ASSERT_EQ(RAY_CASTER_OK, system_prepare(this->System));

  vec3 center = triangle_center(stackScene->faces[1]);
  // origin near to first stack triangle but ray is in opposite direction
  vec3 origin = center + make_vec3(0.f, 0.f, .9f);
  ray_t ray = ray_to_triangle(origin, stackScene->faces[1]);
  face_t* hit_face = 0;
  vec3 hit_point;
  task_t task = { 1, &ray, &hit_face, &hit_point };

  ASSERT_EQ(RAY_CASTER_OK, system_cast(this->System, &task));
  // @todo Provide gtest comparison overloads for vec3
  ASSERT_TRUE(near_enough(hit_point, center));
}

TYPED_TEST(RayCaster, IntersectTriganglesNotPlanes)
{
  scene_t* stackScene = MakeStackScene();

  system_set_scene(this->System, stackScene);
  ASSERT_EQ(RAY_CASTER_OK, system_prepare(this->System));

  vec3 center = triangle_center(stackScene->faces[1]);
  // Ray hit first triangle plane but bypass it by long side
  vec3 origin = center + make_vec3(2.f, 2.f, 2.f);
  ray_t ray = ray_to_triangle(origin, stackScene->faces[1]);
  face_t* hit_face = 0;
  vec3 hit_point;
  task_t task = { 1, &ray, &hit_face, &hit_point };

  ASSERT_EQ(RAY_CASTER_OK, system_cast(this->System, &task));
  // @todo Provide gtest comparison overloads for vec3
  ASSERT_TRUE(near_enough(hit_point, center));
}

TYPED_TEST(RayCaster, ProcessLargeScene)
{
  scene_t largeScene;
  largeScene.n_faces = 1000;
  largeScene.faces = (face_t*)malloc(sizeof(face_t)* largeScene.n_faces);

  face_t reference = make_floor_face1();
  for (int i = -largeScene.n_faces / 2; i != largeScene.n_faces / 2; ++i)
  {
    face_t face = reference;
    vec3 shift = { 0.f, 0.f, .1f * (float)i };
    face.points[0] += shift;
    face.points[1] += shift;
    face.points[2] += shift;
    largeScene.faces[i + largeScene.n_faces / 2] = face;
  }

  system_set_scene(this->System, &largeScene);
  ASSERT_EQ(RAY_CASTER_OK, system_prepare(this->System));

  vec3 center = triangle_center(largeScene.faces[0]);
  // Ray hit first triangle plane but bypass it by long side
  vec3 origin = center + make_vec3(2.f, 2.f, -2.f);
  ray_t ray = ray_to_triangle(origin, largeScene.faces[0]);
  face_t* hit_face = 0;
  vec3 hit_point;
  task_t task = { 1, &ray, &hit_face, &hit_point };

  ASSERT_EQ(RAY_CASTER_OK, system_cast(this->System, &task));
  // @todo Provide gtest comparison overloads for vec3
  ASSERT_TRUE(near_enough(hit_point, center));
}
