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

#include "import_export/obj_import.h"

using namespace testing;

class ObjImport: public Test
{
public:
  ObjImport()
  {
  }
};

TEST_F(ObjImport, ParallelPlanesModel)
{
  form_factors::scene_t* scene = 0;
  ASSERT_EQ(OBJ_IMPORT_OK, obj_import::import_obj("models/parallel_planes.obj", &scene));
  ASSERT_EQ(4, scene->n_faces);
  ASSERT_EQ(2, scene->n_meshes);
}
