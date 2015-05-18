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

#include "import_export/csv_export.h"
#include <fstream>

using namespace testing;

#ifdef _WIN32
#pragma warning(disable:4996)
#endif

class CsvExport : public Test
{
public:
  CsvExport()
  {
  }
};

TEST_F(CsvExport, ParallelPlanesModel)
{
  float factors[4] = { 0, 1, 0, 1 };
  form_factors::task_t task = { 1, factors };
  form_factors::scene_t scene = { 1, 0, 2, 0 };
  ASSERT_EQ(CSV_EXPORT_OK, csv_export::export_csv("test.csv", &scene, &task));
  FILE* f = fopen("test.csv", "r");
  ASSERT_TRUE(f != 0);
  float f0, f1, f2, f3;
  ASSERT_EQ(2, fscanf(f, "%f;%f", &f0, &f1));
  ASSERT_EQ(2, fscanf(f, "%f;%f", &f2, &f3));
  fclose(f);
  remove("test.csv");
}
