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

class RayTraverseGrid
  : public Test
{
public:
  RayTraverseGrid()
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

TEST_F(RayTraverseGrid, HitSingleCell)
{
  ray_t r = { math::make_vec3(1.25f, 1.25f, 0), math::make_vec3(1.5f, 1.5f, 0) };
  coord_set expected = MakeExpected({ { 2, 2 } });
  ASSERT_EQ(expected, Traverse(r));
}

TEST_F(RayTraverseGrid, SmallAngleCollectRow)
{
  ray_t r = { math::make_vec3(0.f, 0.7f, 0), math::make_vec3(2.f, 0.8f, 0) };
  coord_set expected = MakeExpected({ { 0, 1 }, { 1, 1 }, { 2, 1 } });
  ASSERT_EQ(expected, Traverse(r));
}

TEST_F(RayTraverseGrid, VerticalCollectRow)
{
  ray_t r = { math::make_vec3(.7f, 0.1f, 0), math::make_vec3(.7f, 0.2f, 0) };
  coord_set expected = MakeExpected({ { 1, 0 }, { 1, 1 }, { 1, 2 } });
  ASSERT_EQ(expected, Traverse(r));
}

TEST_F(RayTraverseGrid, AlmostCenteredCollectPolylineDiagonal)
{
  ray_t r = { math::make_vec3(.3f, .4f, 0), math::make_vec3(.4f, 0.5f, 0) };
  coord_set expected = MakeExpected({ { 0, 0 }, { 0, 1 }, { 1, 1 }, { 1, 2 }, { 2, 2 } });
  ASSERT_EQ(expected, Traverse(r));
}

TEST_F(RayTraverseGrid, NegativeDirectionCollectCells)
{
  ray_t r = { math::make_vec3(1.4f, 1.49f, 0), math::make_vec3(1.3f, 1.4f, 0) };
  coord_set expected = MakeExpected({ { 0, 0 }, { 0, 1 }, { 1, 1 }, { 1, 2 }, { 2, 2 } });
  ASSERT_EQ(expected, Traverse(r));
}


TEST_F(RayTraverseGrid, NegativeXDirectionCollectCells)
{
  ray_t r = { math::make_vec3(1.1f, 0, 0), math::make_vec3(1, .1f, 0) };
  coord_set expected = MakeExpected({ { 2, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { 0, 2 } });
  ASSERT_EQ(expected, Traverse(r));
}