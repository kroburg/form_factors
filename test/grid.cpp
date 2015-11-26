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

#include "math/grid.h"

#include <set>

using namespace testing;
using namespace math;

class GridTraversal
  : public Test
{
public:
  GridTraversal()
  {
    Grid.base = make_vec3(0, 0, 0);
    Grid.side = make_vec3(0.5f, 0.5f, 0);
    Grid.n_x = 3;
    Grid.n_y = 3;
  }

  typedef std::set<math::grid_coord_t> coord_set;
  coord_set Traverse(ray_t r) const
  {
    coord_set result;
    math::grid_traverse(&Grid, r, (math::grid_traversal_callback)&CellCollector, &result);
    return result;
  }

  coord_set Put(ray_t r) const
  {
    coord_set result;
    math::grid_put(&Grid, r, (math::grid_traversal_callback)&CellCollector, &result);
    return result;
  }

  coord_set Rasterize(triangle_t t) const
  {
    coord_set result;
    math::grid_rasterize(&Grid, t, (math::grid_traversal_callback)&CellCollector, &result);
    return result;
  }

  coord_set MakeExpected(std::initializer_list<std::initializer_list<int> > coords)
  {
    coord_set result;
    for (auto p : coords)
    {
      grid_coord_t c = { *p.begin(), *(p.begin() + 1) };
      result.insert(c);
    }
    return result;
  }

private:
  static bool CellCollector(math::grid_coord_t p, coord_set* coords)
  {
    coords->insert(p);
    return false;
  }

private:
  grid_2d_t Grid;
};

TEST_F(GridTraversal, HitSingleCell)
{
  ray_t r = { math::make_vec3(1.25f, 1.25f, 0), math::make_vec3(1.5f, 1.5f, 0) };
  coord_set expected = MakeExpected({ { 2, 2 } });
  ASSERT_EQ(expected, Traverse(r));
}

TEST_F(GridTraversal, SmallAngleCollectRow)
{
  ray_t r = { math::make_vec3(0.f, 0.7f, 0), math::make_vec3(2.f, 0.8f, 0) };
  coord_set expected = MakeExpected({ { 0, 1 }, { 1, 1 }, { 2, 1 } });
  ASSERT_EQ(expected, Traverse(r));
}

TEST_F(GridTraversal, VerticalCollectRow)
{
  ray_t r = { math::make_vec3(.7f, 0.1f, 0), math::make_vec3(.7f, 0.2f, 0) };
  coord_set expected = MakeExpected({ { 1, 0 }, { 1, 1 }, { 1, 2 } });
  ASSERT_EQ(expected, Traverse(r));
}

TEST_F(GridTraversal, AlmostCenteredCollectPolylineDiagonal)
{
  ray_t r = { math::make_vec3(.3f, .4f, 0), math::make_vec3(.4f, 0.5f, 0) };
  coord_set expected = MakeExpected({ { 0, 0 }, { 0, 1 }, { 1, 1 }, { 1, 2 }, { 2, 2 } });
  ASSERT_EQ(expected, Traverse(r));
}

TEST_F(GridTraversal, NegativeDirectionCollectCells)
{
  ray_t r = { math::make_vec3(1.4f, 1.49f, 0), math::make_vec3(1.3f, 1.4f, 0) };
  coord_set expected = MakeExpected({ { 0, 0 }, { 0, 1 }, { 1, 1 }, { 1, 2 }, { 2, 2 } });
  ASSERT_EQ(expected, Traverse(r));
}


TEST_F(GridTraversal, NegativeXDirectionCollectCells)
{
  ray_t r = { math::make_vec3(1.1f, 0, 0), math::make_vec3(1, .1f, 0) };
  coord_set expected = MakeExpected({ { 2, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { 0, 2 } });
  ASSERT_EQ(expected, Traverse(r));
}

TEST_F(GridTraversal, PutNegativeXYSegment)
{
  ray_t r = { math::make_vec3(.75f, 1.25f, 0), math::make_vec3(.25f, .25f, 0) };
  coord_set expected = MakeExpected({ { 1, 2 }, { 1, 1 }, { 0, 1 }, { 0, 0 } });
  ASSERT_EQ(expected, Put(r));
}

TEST_F(GridTraversal, DontGetCellOnFastSegments)
{
  ray_t r = { math::make_vec3(.55f, .95f, 0), math::make_vec3(.45f, .55f, 0) };
  coord_set expected = MakeExpected({ { 1, 1 }, { 0, 1 } });
  ASSERT_EQ(expected, Put(r));
}

TEST_F(GridTraversal, DontGetCellOnBorderSegment)
{
  ray_t r = { math::make_vec3(.55f, .95f, 0), math::make_vec3(.95f, .55f, 0) };
  coord_set expected = MakeExpected({ { 1, 1 } });
  ASSERT_EQ(expected, Put(r));
}


TEST_F(GridTraversal, RasterizeSmallTriangle)
{
  triangle_t t = { math::make_vec3(.7f, .7f, 0), math::make_vec3(.8f, .8f, 0), math::make_vec3(.8f, .7f, 0) };
  coord_set expected = MakeExpected({ { 1, 1 } });
  ASSERT_EQ(expected, Rasterize(t));
}

TEST_F(GridTraversal, RasterizeLargeTriangle)
{
  triangle_t t = { math::make_vec3(0.75f, 1.25f, 0), math::make_vec3(.25f, .25f, 0), math::make_vec3(1.25f, 0.25f, 0) };
  coord_set expected = MakeExpected({ { 0, 0 }, { 1, 0 }, { 2, 0 }, { 0, 1 }, { 1, 1 }, { 2, 1 }, { 1, 2 } });
  ASSERT_EQ(expected, Rasterize(t));
}

TEST_F(GridTraversal, RasterizeNeedleTriangle)
{
  triangle_t t = { math::make_vec3(0.25f, .75f, 0), math::make_vec3(1.25f, .76f, 0), math::make_vec3(1.25f, 0.74f, 0) };
  coord_set expected = MakeExpected({ { 0, 1 }, { 1, 1 }, { 2, 1 } });
  ASSERT_EQ(expected, Rasterize(t));
}