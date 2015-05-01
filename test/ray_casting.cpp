#include "gtest/gtest.h"

#include "ray_caster/system.h"

using namespace ray_caster;
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
    system_free(System);
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
  scene_t* emptyScence = MakeEmptyScene();
  ASSERT_EQ(RAY_CASTER_OK, system_set_scene(System, emptyScence));
}

TYPED_TEST(RayCaster, AcceptFloorScene)
{
  scene_t* floorScene = MakeFloorScene();
  ASSERT_EQ(RAY_CASTER_OK, system_set_scene(System, floorScene));
}

TYPED_TEST(RayCaster, PrepareFailsForNoScene)
{ 
  ASSERT_EQ(-RAY_CASTER_ERROR, system_prepare(System));
}

TYPED_TEST(RayCaster, PrepareFailsForEmptyScene)
{
  scene_t* emptyScence = MakeEmptyScene();
  system_set_scene(System, emptyScence);
  ASSERT_EQ(-RAY_CASTER_ERROR, system_prepare(System));
}

TYPED_TEST(RayCaster, PreparePassForNotEmptyScene)
{
  scene_t* floorScene = MakeFloorScene();
  system_set_scene(System, floorScene);
  ASSERT_EQ(RAY_CASTER_OK, system_prepare(System));
}