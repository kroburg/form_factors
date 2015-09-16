// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
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

#include "obj_export.h"

namespace obj_export
{
  void export_vertex(FILE* out, const math::vec3& p)
  {
    fprintf(out, "v %f %f %f\n", p.x, p.y, p.z);
  }

  void export_vertecies(FILE* out, const subject::scene_t* scene)
  {
    int i = 0;
    const subject::face_t* f = scene->faces;
    for (; i != scene->n_faces; ++i, ++f)
    { 
      export_vertex(out, f->points[0]);
      export_vertex(out, f->points[1]);
      export_vertex(out, f->points[2]);
    }
    fprintf(out, "\n");
  }

  void export_materials(FILE* out, const subject::scene_t* scene)
  {
    int i = 0;
    const subject::material_t* m = scene->materials;
    for (; i != scene->n_materials; ++i, ++m)
    {
      fprintf(out, "newmtl %s\n", m->name);
      fprintf(out, "shell.density %f\n", m->shell.density);
      fprintf(out, "shell.heat_capacity %f\n", m->shell.heat_capacity);
      fprintf(out, "shell.thermal_conductivity %f\n", m->shell.thermal_conductivity);
      fprintf(out, "shell.thickness %f\n", m->shell.thickness);
      fprintf(out, "front.specular_reflectance %f\n", m->front.specular_reflectance);
      fprintf(out, "front.diffuse_reflectance %f\n", m->front.diffuse_reflectance);
      fprintf(out, "front.absorbance %f\n", m->front.absorbance);
      fprintf(out, "front.transmittance %f\n", m->front.transmittance);
      fprintf(out, "front.emissivity %f\n", m->front.emissivity);
      fprintf(out, "rear.specular_reflectance %f\n", m->rear.specular_reflectance);
      fprintf(out, "rear.diffuse_reflectance %f\n", m->rear.diffuse_reflectance);
      fprintf(out, "rear.absorbance %f\n", m->rear.absorbance);
      fprintf(out, "rear.transmittance %f\n", m->rear.transmittance);
      fprintf(out, "rear.emissivity %f\n", m->rear.emissivity);
      fprintf(out, "\n");
    }
  }

  void export_meshes(FILE* out, const subject::scene_t* scene)
  { 
    const subject::mesh_t* m = scene->meshes;
    for (int i = 0; i != scene->n_meshes; ++i, ++m)
    { 
      fprintf(out, "g mesh_%d\n", i);
      fprintf(out, "usemtl %s\n", scene->materials[m->material_idx].name);
      
      const subject::face_t* f = &scene->faces[m->first_idx];
      for (int j = 0; j != m->n_faces; ++j, ++f)
      {
        int vertex_idx = 1 + 3 * (m->first_idx + j);
        fprintf(out, "f %d %d %d\n", vertex_idx, vertex_idx + 1, vertex_idx + 2);
      }
      fprintf(out, "\n");
    }
  }

  int scene(FILE* out, const subject::scene_t* scene)
  {
    export_vertecies(out, scene);
    export_materials(out, scene);
    export_meshes(out, scene);
    return 0;
  }

  int task(FILE* out, int n_meshes, const thermal_solution::task_t* task)
  {
    fprintf(out, "newfrm %d\n", task->n_step);
    fprintf(out, "time_step %f\n", task->time_delta);
    fprintf(out, "temperatures");
    for (const float* f = task->temperatures; f != task->temperatures + n_meshes; ++f)
    {
      fprintf(out, " %f", *f);
    }
    fprintf(out, "\n\n");
    return 0;
  }
}