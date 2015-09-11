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

#include <math/operations.h>
#include <math/triangle.h>

using namespace testing;
using namespace math;

const vec3 Epsilon = { FLT_EPSILON, FLT_EPSILON, FLT_EPSILON };
const vec3 A = { 0.f, 0.f, 0.f };
const vec3 B = { 1.f, 0.f, 0.f };
const vec3 C = { 0.f, .5f, 0.f };
const vec3 D = { 1.f, .5f, 0.f };
const vec3 E = (D + B) / 2.f;
const vec3 F = (D + C) / 2.f;

TEST(FindLeastSide, ScanAllEdges)
{
  triangle_t t = make_face(A, B, C);
  float least_side = triangle_least_side_square(t);
  EXPECT_NEAR(0.25f, least_side, 0.01);

  t = make_face(C, A, B);
  least_side = triangle_least_side_square(t);
  EXPECT_NEAR(0.25f, least_side, 0.01);

  t = make_face(B, C, A);
  least_side = triangle_least_side_square(t);
  EXPECT_NEAR(0.25f, least_side, 0.01);
};

TEST(FindAdjacentVertex, ScanAllVertex)
{
  triangle_t t = make_face(A, B, C);
  EXPECT_EQ(0, triangle_find_adjacent_vertex(t, A + Epsilon));
  EXPECT_EQ(1, triangle_find_adjacent_vertex(t, B + Epsilon));
  EXPECT_EQ(2, triangle_find_adjacent_vertex(t, C + Epsilon));
};

TEST(FindAdjacentVertex, ConsiderTriangleScale)
{
  triangle_t t = make_face(A * FLT_EPSILON, B * FLT_EPSILON, C * FLT_EPSILON);
  EXPECT_EQ(-1, triangle_find_adjacent_vertex(t, A + Epsilon));
  EXPECT_EQ(-1, triangle_find_adjacent_vertex(t, B + Epsilon));
  EXPECT_EQ(-1, triangle_find_adjacent_vertex(t, C + Epsilon));
};

TEST(FindAdjacentVertex, MapTriangles)
{
  triangle_t t1 = make_face(A, B, C);
  triangle_t t2 = make_face(C, A, B);
  int actual = triangle_find_adjacent_vertices(t1, t2);

  /*
    l[2] ? r[2], l[2] ? r[1], l[2] ? r[0],
    l[1] ? r[2], l[1] ? r[1], l[1] ? r[0],
    l[0] ? r[2], l[0] ? r[1], l[0] ? r[0]

    0 0 1 // C
    1 0 0 // B
    0 1 0 // A
  */

  int expected = 1 << 6 | 1 << 5 | 1 << 1;
  EXPECT_EQ(expected, actual);
};

TEST(NormalsDirectivity, UnidirectionalForSameTriangle)
{
  int mapping = triangle_find_adjacent_vertices(make_face(A, D, B), make_face(A, D, B));
  EXPECT_TRUE(math::triangle_has_unidirectrional_normals(mapping));
}

TEST(NormalsDirectivity, UnidirectionalForADB_ACD)
{
  int mapping = triangle_find_adjacent_vertices(make_face(A, D, B), make_face(A, C, D));
  EXPECT_TRUE(math::triangle_has_unidirectrional_normals(mapping));
}

TEST(NormalsDirectivity, ContradictionalForADB_ADC)
{
  int mapping = triangle_find_adjacent_vertices(make_face(A, D, B), make_face(A, D, C));
  EXPECT_FALSE(math::triangle_has_unidirectrional_normals(mapping));
}

TEST(HasAdjacentEdge, IgnoreUnrelatedTriangles)
{
  triangle_t t1 = make_face(A, B, C);
  triangle_t t2 = make_face(D, E, F);
  EXPECT_FALSE(triangle_has_adjacent_edge(t1, t2));
}

TEST(HasAdjacentEdge, IgnoreTriangleWithSingleCommonVertex)
{
  triangle_t t1 = make_face(A, B, C);
  triangle_t t2 = make_face(A, E, F);
  EXPECT_FALSE(triangle_has_adjacent_edge(t1, t2));
}

TEST(HasAdjacentEdge, AcceptCommonEdge)
{
  triangle_t t1 = make_face(A, B, C);
  triangle_t t2 = make_face(B, C, D);
  EXPECT_TRUE(triangle_has_adjacent_edge(t1, t2));
}

TEST(HasAdjacentEdge, AcceptMultipleCommonEdges)
{
  triangle_t t1 = make_face(A, B, C);
  triangle_t t2 = make_face(C, A, B);
  EXPECT_TRUE(triangle_has_adjacent_edge(t1, t2));
}

