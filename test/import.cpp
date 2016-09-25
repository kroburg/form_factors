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

TEST(ObjImport, ParseFaces)
{
  subject::scene_t* scene = 0;
  ASSERT_EQ(OBJ_IMPORT_OK, obj_import::scene("models/parallel_planes.obj", &scene));
  ASSERT_EQ(4, scene->n_faces);
}

TEST(ObjImport, ParseMeshes)
{
  subject::scene_t* scene = 0;
  ASSERT_EQ(OBJ_IMPORT_OK, obj_import::scene("models/parallel_planes.obj", &scene));
  ASSERT_EQ(2, scene->n_meshes);
}

TEST(ObjImport, DetectUnknownMaterial)
{
  subject::scene_t* scene = 0;
  ASSERT_EQ(-OBJ_IMPORT_MATERIAL_NOT_DEFINED, obj_import::scene("models/unknown_material.obj", &scene));
}

TEST(ObjImport, ParseMaterials)
{
  subject::scene_t* scene = 0;
  ASSERT_EQ(OBJ_IMPORT_OK, obj_import::scene("models/test_material.obj", &scene));
  ASSERT_EQ(1, scene->n_materials);
  subject::material_t& m = scene->materials[0];
  ASSERT_NEAR(1, m.shell.density, 0.01);
  ASSERT_NEAR(2, m.shell.heat_capacity, 0.01);
  ASSERT_NEAR(3, m.shell.thermal_conductivity, 0.01);
  ASSERT_NEAR(4, m.shell.thickness, 0.01);
  ASSERT_NEAR(1.1, m.front.specular_reflectance, 0.01);
  ASSERT_NEAR(1.2, m.front.diffuse_reflectance, 0.01);
  ASSERT_NEAR(1.3, m.front.absorbance, 0.01);
  ASSERT_NEAR(1.4, m.front.transmittance, 0.01);
  ASSERT_NEAR(1.5, m.front.emissivity, 0.01);
  ASSERT_NEAR(2.1, m.rear.specular_reflectance, 0.01);
  ASSERT_NEAR(2.2, m.rear.diffuse_reflectance, 0.01);
  ASSERT_NEAR(2.3, m.rear.absorbance, 0.01);
  ASSERT_NEAR(2.4, m.rear.transmittance, 0.01);
  ASSERT_NEAR(2.5, m.rear.emissivity, 0.01);
}
