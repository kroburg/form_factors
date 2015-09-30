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

#include <subject/system.h>
#include <math/operations.h>

using namespace testing;
using namespace subject;

const math::vec3 A = { 0.f, 0.f, 0.f };
const math::vec3 B = { 1.f, 0.f, 0.f };
const math::vec3 C = { 0.f, 1.f, 0.f };
const math::vec3 D = { 1.f, 1.f, 0.f };
const math::vec3 E = { 0.f, -1.f, 0.f };

struct LinearSearch
{
  int walk(const face_t* faces, int n_faces, face_graph_walker walker, void* param)
  {
    return face_walk_graph_n2c(faces, n_faces, walker, param);
  }
};

struct IndexedSearch
{
  int walk(const face_t* faces, int n_faces, face_graph_walker walker, void* param)
  {
    face_graph_index_t* index = face_graph_index_create(faces, n_faces);
    int r = face_graph_walk_index(index, walker, param);
    face_graph_index_free(index);
    return r;
  }
};

template <typename EngineType>
class MeshGraph
   : public Test
   , public EngineType
{
};

typedef ::testing::Types<LinearSearch, IndexedSearch> SearchEngineTypes;
TYPED_TEST_CASE(MeshGraph, SearchEngineTypes);

int ReportSingleFaceCheck(int current_idx, int leaf_idx, int mapping, bool have_more, void* param)
{
  EXPECT_TRUE(param != 0);

  EXPECT_EQ(0, current_idx);
  EXPECT_EQ(-1, leaf_idx);
  EXPECT_EQ(false, have_more);
  
  ++*(int*)param;
  return 0;
}

TYPED_TEST(MeshGraph, ReportSingleFace)
{
  face_t faces[1] = { make_face(A, B, C) };

  int invoke_counter = 0;
  this->walk(faces, sizeof(faces) / sizeof(face_t), ReportSingleFaceCheck, &invoke_counter);
  ASSERT_EQ(1, invoke_counter);
};

int ReportSecondFaceCheck(int current_idx, int leaf_idx, int mapping, bool have_more, void* param)
{
  EXPECT_TRUE(param != 0);
  if (param == 0)
    return -1;

  switch (++*(int*)param)
  {
  case 1:
  {
    EXPECT_EQ(0, current_idx);
    EXPECT_EQ(1, leaf_idx);
    EXPECT_EQ(false, have_more);
  }
  break;

  case 2:
  {
    EXPECT_EQ(1, current_idx);
    EXPECT_EQ(-1, leaf_idx);
    EXPECT_EQ(false, have_more);
  }
  break;
  }

  return 0;
}

TYPED_TEST(MeshGraph, ReportSecondFace)
{
  face_t faces[2] = { make_face(A, B, C), make_face(B, C, D) };
 
  int invoke_counter = 0;
  this->walk(faces, sizeof(faces) / sizeof(face_t), ReportSecondFaceCheck, &invoke_counter);
  ASSERT_EQ(2, invoke_counter);
};

int SignalHaveMoreForMultipleFacesCheck(int current_idx, int leaf_idx, int mapping, bool have_more, void* param)
{
  EXPECT_TRUE(param != 0);
  if (param == 0)
    return -1;

  switch (++*(int*)param)
  {
  case 1:
  {
    EXPECT_EQ(0, current_idx);
    EXPECT_EQ(1, leaf_idx);
    EXPECT_EQ(true, have_more);
  }
  break;

  case 2:
  {
    EXPECT_EQ(0, current_idx);
    EXPECT_EQ(2, leaf_idx);
    EXPECT_EQ(false, have_more);
  }
  break;

  case 3:
  {
    EXPECT_EQ(1, current_idx);
    EXPECT_EQ(-1, leaf_idx);
    EXPECT_EQ(false, have_more);
  }
  break;

  case 4:
  {
    EXPECT_EQ(2, current_idx);
    EXPECT_EQ(-1, leaf_idx);
    EXPECT_EQ(false, have_more);
  }
  break;
  }

  return 0;
}

TYPED_TEST(MeshGraph, SignalHaveMoreForMultipleFaces)
{
  face_t faces[3] = { make_face(A, B, C), make_face(B, C, D), make_face(A, B, E) };

  int invoke_counter = 0;
  this->walk(faces, sizeof(faces) / sizeof(face_t), SignalHaveMoreForMultipleFacesCheck, &invoke_counter);
  ASSERT_EQ(4, invoke_counter);
};

int ReportDisconnectedFacesCheck(int current_idx, int leaf_idx, int mapping, bool have_more, void* param)
{
  EXPECT_TRUE(param != 0);
  if (param == 0)
    return -1;

  switch (++*(int*)param)
  {
  case 1:
  {
    EXPECT_EQ(0, current_idx);
    EXPECT_EQ(-1, leaf_idx);
    EXPECT_EQ(false, have_more);
  }
  break;

  case 2:
  {
    EXPECT_EQ(1, current_idx);
    EXPECT_EQ(-1, leaf_idx);
    EXPECT_EQ(false, have_more);
  }
  break;
  }

  return 0;
}

TYPED_TEST(MeshGraph, ReportDisconnectedFaces)
{
  face_t faces[2] = { make_face(B, C, D), make_face(A, B, E) };

  int invoke_counter = 0;
  this->walk(faces, sizeof(faces) / sizeof(face_t), ReportDisconnectedFacesCheck, &invoke_counter);
  ASSERT_EQ(2, invoke_counter);
};

int StopOnDemandCheck(int current_idx, int leaf_idx, int mapping, bool have_more, void* param)
{
  EXPECT_TRUE(param != 0);
  if (param == 0)
    return -1;

  ++*(int*)param;
  
  return 42;
}

TYPED_TEST(MeshGraph, StopOnDemand)
{
  face_t faces[2] = { make_face(B, C, D), make_face(A, B, E) };

  int invoke_counter = 0;
  int return_code = this->walk(faces, sizeof(faces) / sizeof(face_t), StopOnDemandCheck, &invoke_counter);
  ASSERT_EQ(1, invoke_counter);
  ASSERT_EQ(42, return_code);
};