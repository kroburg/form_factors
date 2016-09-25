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

/**
 * This module contains GPU-oriented methods to cast rays on Cuda devices.
 * Also module contains ray_caster::system_t Cuda implementation.
 */

#include "system.cuh"
#include "cuda_system.h"

#include "../math/grid.h"
#include "../math/triangle.h"

#include <helper_cuda.h>
#include <helper_math.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#ifndef _WIN32
#include <values.h>
#endif
#include <assert.h>

namespace naive_cuda_ray_caster
{
  /// @brief Extended base system_t (C-style polymorphism)
  struct cuda_system_t : ray_caster::system_t
  {
    ray_caster::scene_t* scene;
    int n_faces;
    bb_face_t* faces;
    int dev_id;
    int n_tpb;
  };

  /// @brief Initializes system after creation, finds Cuda device.
  int init(cuda_system_t* system)
  {
    system->scene = 0;
    system->n_faces = 0;
    system->faces = 0;
    const char* argv[] = { "" };
    system->dev_id = findCudaDevice(1, (const char **)argv);
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, system->dev_id));
    //system->n_tpb = deviceProp.maxThreadsPerBlock;
    system->n_tpb = 128;

    return RAY_CASTER_OK;
  }

  /// @brief Shutdowns system prior to free memory.
  int shutdown(cuda_system_t* system)
  {
    if (system->faces)
      cudaFree(system->faces);
    system->scene = 0;
    cudaDeviceReset();
    return RAY_CASTER_OK;
  }

  /// @brief Sets loaded scene (polygons in meshes) for ray caster.
  int set_scene(cuda_system_t* system, ray_caster::scene_t* scene)
  {
    if (system->faces)
    {
      cudaFree(system->faces);
      system->faces = 0;
      system->n_faces = 0;
    }

    system->scene = scene;
    system->n_faces = scene->n_faces;

    if (system->n_faces == 0)
      /// @todo: May be -RAY_CASTER_ERROR?
      return RAY_CASTER_OK;

    // Allocate Cuda memory for scene's faces.
    ray_caster::face_t* sourceFaces;
    checkCudaErrors(cudaMalloc((void**)&sourceFaces, scene->n_faces * sizeof(ray_caster::face_t)));
    checkCudaErrors(cudaMemcpy(sourceFaces, scene->faces, sizeof(ray_caster::face_t) * scene->n_faces, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&system->faces, scene->n_faces * sizeof(bb_face_t)));
    
    // @todo How to test that faces upload working?
    // @todo Decide on "perfect occupancy".
    int n_tpb = system->n_tpb; // Threads per block
    int n_blocks = (system->n_faces + n_tpb - 1) / n_tpb; // Total number of blocks
    load_scene_faces << <n_blocks, n_tpb >> >(sourceFaces, system->faces, system->n_faces);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(sourceFaces));

    return RAY_CASTER_OK;
  }

  /// @brief Checks system consistency before ray casting.
  int prepare(cuda_system_t* system)
  {
    if (system->n_faces == 0)
      return -RAY_CASTER_ERROR;
    return RAY_CASTER_OK;
  }

  // Convertion from and to device's types.

  vec3 ext2loc(math::vec3 a)
  {
    return make_float3(a.x, a.y, a.z);
  }

  math::vec3 loc2ext(vec3 a)
  {
    return math::make_vec3(a.x, a.y, a.z);
  }

  /// @brief Distance to intersection of given ray with face in faces array with face_idx index.
  struct distance_reduce_step_t
  {
    float distance;
    int face_idx;
  };

  /// @brief Casts rays of given task task for given scene.
  int cast(cuda_system_t* system, ray_caster::task_t* task)
  {
    using namespace ray_caster;

    const int n_rays = task->n_tasks;

    const int max_concurrent_rays = (int)pow(2, ceil(log2(16384.f * 1024 / (system->n_faces + 1023))));

    thrust::device_vector<vec3> points(max_concurrent_rays);
    thrust::device_vector<int> indices(max_concurrent_rays);
    thrust::host_vector<vec3> h_points;
    thrust::host_vector<int> h_indices;

    bb_ray_t* d_rays;
    // Allocating Cuda arrays by blocks
    checkCudaErrors(cudaMalloc((void**)&d_rays, n_rays * sizeof(bb_ray_t)));
    {
      thrust::device_vector<math::ray_t> client_rays(n_rays);
      checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(client_rays.data()), task->ray, n_rays * sizeof(math::ray_t), cudaMemcpyHostToDevice));
      int n_tpb = system->n_tpb;
      int n_blocks = (n_rays + n_tpb - 1) / n_tpb;
      load_rays << <n_blocks, n_tpb >> >(thrust::raw_pointer_cast(client_rays.data()), d_rays, n_rays);
      checkCudaErrors(cudaPeekAtLastError());
      checkCudaErrors(cudaDeviceSynchronize());
    }

    
    for (int t = 0; t < task->n_tasks;)
    {
      int n_rays = t + max_concurrent_rays < task->n_tasks ? max_concurrent_rays : task->n_tasks - t;

      int n_tpb = system->n_tpb < system->n_faces ? system->n_tpb : system->n_faces;
      int shared_size = n_tpb * sizeof(distance_reduce_step_t);
      dim3 threads(1, n_tpb);
      dim3 grid(n_rays, 1);
      cast_scene_faces_with_reduction << <grid, threads, shared_size >> >(system->faces, system->n_faces, d_rays + t, thrust::raw_pointer_cast(indices.data()), thrust::raw_pointer_cast(points.data()));
      checkCudaErrors(cudaPeekAtLastError());
      checkCudaErrors(cudaDeviceSynchronize());

      h_points = points;
      h_indices = indices;

      for (int j = 0; j != n_rays; ++j)
      {
        if (h_indices[j] != -1)
        {
          task->hit_face[t + j] = system->scene->faces + h_indices[j];
          task->hit_point[t + j] = loc2ext(h_points[j]);
        }
        else
        {
          task->hit_face[t + j] = 0;
        }
      }

      t += n_rays;
    }

    checkCudaErrors(cudaFree(d_rays));

    return RAY_CASTER_OK;
  }

  /// @brief Creates virtual methods table from local methods.
  const ray_caster::system_methods_t methods =
  {
    (int(*)(ray_caster::system_t* system))&init,
    (int(*)(ray_caster::system_t* system))&shutdown,
    (int(*)(ray_caster::system_t* system, ray_caster::scene_t* scene))&set_scene,
    (int(*)(ray_caster::system_t* system))&prepare,
    (int(*)(ray_caster::system_t* system, ray_caster::task_t* task))&cast,
  };

  /// @brief Creates base system for ray caster.
  ray_caster::system_t* system_create()
  {
    cuda_system_t* s = (cuda_system_t*)malloc(sizeof(cuda_system_t));
    s->methods = &methods;
    return s;
  }

  /// @brief Creates bounding box for triangle.
  __device__ void init_face_bbox(bb_face_t* f)
  {
    f->bbox[0] = fminf(fminf(f->points[0], f->points[1]), f->points[2]);
    f->bbox[1] = fmaxf(fmaxf(f->points[0], f->points[1]), f->points[2]);
  }

  /// @brief Loads scene to GPU and converts base face_t to cuda_ray_caster::face_t (with bounding box).
   __global__ void load_scene_faces(const ray_caster::face_t* source, bb_face_t* target, int n_faces)
  {
    // Copy block-by-block.
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_faces)
    {
      // @todo Implement some kind of assertion here.
      //assert(sizeof(face_t::points) == sizeof(ray_caster::face_t::points));
      memcpy(target[i].points, source[i].points, sizeof(bb_face_t::points));
      init_face_bbox(&target[i]);
    }
  }

  /// @brief Loads task with n_rays to GPU.
   __global__ void load_rays(const math::ray_t* source, bb_ray_t* target, int n_rays)
  {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n_rays)
    {
      memcpy(&target[i], &source[i], 2 * sizeof(vec3));
      target[i].inv_dir = make_float3(1.) / (target[i].direction - target[i].origin);
    }
  }

  /// @brief Finds intersection of given ray (blockIdx.x) with n_faces faces (y block dimension).
  __device__ void cast_scene_intersection_step(const bb_face_t* faces, int n_faces, const bb_ray_t* rays, int* indices, vec3* points)
  {
    extern __shared__ distance_reduce_step_t distances[];

    // Current intersection distance for given ray and block's faces.
    distance_reduce_step_t& step = distances[threadIdx.y];
    step.distance = FLT_MAX;
    step.face_idx = -1;

    int face_idx = threadIdx.y;
    const int ray_idx = blockIdx.x;

    // Intersection step.
    for (; face_idx < n_faces; face_idx += blockDim.y)
    { 
      bb_ray_t ray = rays[ray_idx];
      // If ray intersected with bounding box.
      if (face_bbox_intersect(ray, &faces[face_idx]))
      {
        vec3 point;
        int result_code = cuda_math::triangle_intersect(ray.origin, ray.direction, faces[face_idx].points, &point);
        if (result_code == TRIANGLE_INTERSECTION_UNIQUE)
        {
          // Final intersection check.
          point -= ray.origin;
          float distance = dot(point, point);
          // If less then other faces - write new distance.
          if (distance < step.distance)
          {
            step.distance = distance;
            step.face_idx = face_idx;
          }
        }
      }
    }

    // Reduction cycle.
    __syncthreads();
  }

  /// @brief Reduces all distances for ray (blockIdx.x) and all faces.
  __device__ void cast_scene_reduction_step(const bb_face_t* faces, int n_faces, const bb_ray_t* rays, int* indices, vec3* points)
  { 
    extern __shared__ distance_reduce_step_t distances[];

    int thread_idx = threadIdx.y;
    int m = blockDim.y;
    
    while (m > 1)
    {
      int half_m = m >> 1;
      if (thread_idx < half_m)
      {
        if (distances[thread_idx].distance > distances[m - thread_idx - 1].distance)
        {
          distances[thread_idx].distance = distances[m - thread_idx - 1].distance;
          distances[thread_idx].face_idx = distances[m - thread_idx - 1].face_idx;
        }
      }
      __syncthreads();

      m -= half_m;
    }
  }

  __global__ void cast_scene_faces_with_reduction(const bb_face_t* faces, int n_faces, const bb_ray_t* rays, int* indices, vec3* points)
  {
    extern __shared__ distance_reduce_step_t distances[];

    int thread_idx = threadIdx.y;
    int ray_idx = blockIdx.x;

    cast_scene_intersection_step(faces, n_faces, rays, indices, points);
    cast_scene_reduction_step(faces, n_faces, rays, indices, points);
    // Finalization step
    if (thread_idx == 0)
    {
      bb_ray_t ray = rays[ray_idx];
      int face_idx = distances[0].face_idx;
      if (face_idx != -1)
      {
        int result_code = cuda_math::triangle_intersect(ray.origin, ray.direction, faces[face_idx].points, &points[ray_idx]);
        indices[ray_idx] = result_code == TRIANGLE_INTERSECTION_UNIQUE ? distances[0].face_idx : -1;
      }
      else
      {
        indices[ray_idx] = -1;
      }      
    }
  }

  __device__ bool face_bbox_intersect(bb_ray_t ray, const bb_face_t* face)
  { 
    const vec3 boxMin = face->bbox[0] - ray.origin;
    const vec3 boxMax = face->bbox[1] - ray.origin;

    float lo = ray.inv_dir.x*boxMin.x;
    float hi = ray.inv_dir.x*boxMax.x;

    float tmin, tmax;
    tmin = fminf(lo, hi);
    tmax = fmaxf(lo, hi);

    float lo1 = ray.inv_dir.y*boxMin.y;
    float hi1 = ray.inv_dir.y*boxMax.y;

    tmin = fmaxf(tmin, fminf(lo1, hi1));
    tmax = fminf(tmax, fmaxf(lo1, hi1));

    float lo2 = ray.inv_dir.z*boxMin.z;
    float hi2 = ray.inv_dir.z*boxMax.z;

    tmin = fmaxf(tmin, fminf(lo2, hi2));
    tmax = fminf(tmax, fmaxf(lo2, hi2));

    return (tmin <= tmax) && (tmax > 0.f);
  }  
}

namespace zgrid_cuda_ray_caster
{
  struct cuda_system_t : ray_caster::system_t
  {
    ray_caster::scene_t* scene;
    int n_faces;
    face_t* faces;
    grid_t* grid;
    grid_cell_t* grid_cells;
    int* grid_triangles;
    int dev_id;
    int n_tpb;
  };

  /// @brief Initializes system after creation, finds Cuda device.
  int init(cuda_system_t* system)
  {
    system->scene = 0;
    system->n_faces = 0;
    system->faces = 0;
    system->grid = 0;
    system->grid_cells = 0;
    system->grid_triangles = 0;
    const char* argv[] = { "" };
    system->dev_id = findCudaDevice(1, (const char **)argv);
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, system->dev_id));
    system->n_tpb = deviceProp.maxThreadsPerBlock;
    system->n_tpb = 32;

    return RAY_CASTER_OK;
  }

  void release_volatile_data(cuda_system_t* system)
  {
    if (system->faces)
      cudaFree(system->faces);
    if (system->grid)
      cudaFree(system->grid);
    if (system->grid_cells)
      cudaFree(system->grid_cells);
    if (system->grid_triangles)
      cudaFree(system->grid_triangles);

    system->faces = 0;
    system->n_faces = 0;
    system->grid = 0;
    system->grid_cells = 0;
    system->grid_triangles = 0;
  }

  int shutdown(cuda_system_t* system)
  {
    release_volatile_data(system);
    cudaDeviceReset();
    return RAY_CASTER_OK;
  }

  int set_scene(cuda_system_t* system, ray_caster::scene_t* scene)
  {
    release_volatile_data(system);

    system->scene = scene;
    system->n_faces = scene->n_faces;

    return RAY_CASTER_OK;
  }

  vec3 ext2loc(math::vec3 a)
  {
    return make_float3(a.x, a.y, a.z);
  }

  math::vec3 loc2ext(vec3 a)
  {
    return math::make_vec3(a.x, a.y, a.z);
  }

  /// @brief Checks system consistency before ray casting.
  int prepare(cuda_system_t* system)
  {
    if (system->n_faces == 0)
      return -RAY_CASTER_ERROR;

    ray_caster::scene_t* scene = system->scene;

    // Allocate Cuda memory for scene's faces.
    assert(sizeof(face_t) == sizeof(ray_caster::face_t));
    checkCudaErrors(cudaMalloc((void**)&system->faces, scene->n_faces * sizeof(ray_caster::face_t)));
    checkCudaErrors(cudaMemcpy(system->faces, scene->faces, sizeof(ray_caster::face_t) * scene->n_faces, cudaMemcpyHostToDevice));
    
    math::triangles_analysis_t stat = triangles_analyze(system->scene->faces, system->scene->n_faces);
    math::grid_2d_t grid = grid_deduce_optimal(stat);
    math::grid_2d_index_t* index = grid_make_index(&grid);
    // @todo Make kernel for indexing
    grid_index_triangles(&grid, index, system->scene->faces, system->scene->n_faces);
    
    checkCudaErrors(cudaMalloc((void**)&system->grid_cells, grid.n_x * grid.n_y * sizeof(grid_cell_t)));
    checkCudaErrors(cudaMemcpy(system->grid_cells, index->cells, grid.n_x * grid.n_y * sizeof(grid_cell_t), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&system->grid_triangles, index->n_triangles * sizeof(int)));
    checkCudaErrors(cudaMemcpy(system->grid_triangles, index->triangles, index->n_triangles * sizeof(int), cudaMemcpyHostToDevice));

    grid_t device_grid;
    device_grid.base = ext2loc(grid.base);
    device_grid.side = ext2loc(grid.side);
    device_grid.size = ext2loc(grid.size);
    device_grid.n_x = grid.n_x;
    device_grid.n_y = grid.n_y;
    device_grid.cells = system->grid_cells;
    device_grid.triangles = system->grid_triangles;

    checkCudaErrors(cudaMalloc((void**)&system->grid, sizeof(grid_t)));
    checkCudaErrors(cudaMemcpy(system->grid, &device_grid, sizeof(grid_t), cudaMemcpyHostToDevice));

    grid_free_index(index);
    
    return RAY_CASTER_OK;
  }

  /// @brief Casts rays of given task task for given scene.
  int cast(cuda_system_t* system, ray_caster::task_t* task)
  {
    const int n_rays = task->n_tasks;

    ray_t* rays;
    checkCudaErrors(cudaMalloc((void**)&rays, n_rays * sizeof(ray_t)));
    checkCudaErrors(cudaMemcpy(rays, task->ray, n_rays * sizeof(ray_t), cudaMemcpyHostToDevice));

    const int max_concurrent_rays = 1 << 10;

    thrust::device_vector<vec3> points(max_concurrent_rays);
    thrust::device_vector<int> indices(max_concurrent_rays);
    thrust::host_vector<vec3> h_points;
    thrust::host_vector<int> h_indices;

    for (int t = 0; t < task->n_tasks;)
    {
      int n_rays = t + max_concurrent_rays < task->n_tasks ? max_concurrent_rays : task->n_tasks - t;
      int n_blocks = (n_rays + system->n_tpb - 1) / system->n_tpb;

      dim3 threads(system->n_tpb);
      dim3 grid(n_blocks);
      cast_scene << <grid, threads>> >(system->faces, system->grid, rays + t, n_rays, thrust::raw_pointer_cast(indices.data()), thrust::raw_pointer_cast(points.data()));
      checkCudaErrors(cudaPeekAtLastError());
      checkCudaErrors(cudaDeviceSynchronize());

      h_points = points;
      h_indices = indices;

      for (int j = 0; j != n_rays; ++j)
      {
        if (h_indices[j] != -1)
        {
          task->hit_face[t + j] = system->scene->faces + h_indices[j];
          task->hit_point[t + j] = loc2ext(h_points[j]);
        }
        else
        {
          task->hit_face[t + j] = 0;
        }
      }

      t += n_rays;
    }

    cudaFree(rays);
    return RAY_CASTER_OK;
  }

  __device__ ray_t make_ray(const vec3& origin, const vec3& direction)
  {
    return{ origin, direction };
  }

  __device__ bool ray_intersect_segment(ray_t p, ray_t q, float& s)
  {
    vec3 v = q.direction - q.origin;
    vec3 w = p.origin - q.origin;
    vec3 u = p.direction - p.origin;

    float d = (v.x * u.y - v.y * u.x);

    s = (v.y * w.x - v.x * w.y) / d;
    float t = (u.x * w.y - u.y * w.x) / -d;

    return t >= 0 && t <= 1 && s >= 0;
  }

  __device__ bool rebase_if_out_of_bound(const grid_t* grid, ray_t& ray)
  {
    vec3 u = ray.origin - grid->base;
    vec3 h = u - grid->size;
    vec3 v = ray.direction - ray.origin;

    bool out_of_bound = u.x < 0 || h.x > 0 || u.y < 0 || h.y > 0;
    if (!out_of_bound)
      return true;

    ray_t plane_ray = make_ray(u, ray.direction - grid->base);

    float s1;
    float s2;
    float s3;
    float s4;
    bool i1 = ray_intersect_segment(plane_ray, make_ray(make_float3(0, 0, 0), make_float3(grid->size.x, 0, 0)), s1);
    bool i2 = ray_intersect_segment(plane_ray, make_ray(make_float3(0, 0, 0), make_float3(0, grid->size.y, 0)), s2);
    bool i3 = ray_intersect_segment(plane_ray, make_ray(make_float3(0, grid->size.y, 0), make_float3(grid->size.x, grid->size.y, 0)), s3);
    bool i4 = ray_intersect_segment(plane_ray, make_ray(make_float3(grid->size.x, 0, 0), make_float3(grid->size.x, grid->size.y, 0)), s4);

    if (!i1 && !i2 && !i3 && !i4)
      return false;

    float s = FLT_MAX;
    if (i1) s = fminf(s, s1);
    if (i2) s = fminf(s, s2);
    if (i3) s = fminf(s, s3);
    if (i4) s = fminf(s, s4);

    vec3 shift = v * (s + 0.00001f);
    ray.origin += shift;
    ray.direction += shift;

    return true;
  }

  struct callback_param_t
  {
    const face_t* faces;
    const grid_t* grid;
    ray_t ray;
    int* hit_face;
    vec3* hit_point;
  };

  struct grid_coord_t
  {
    int x;
    int y;
  };

  /// @brief Finds intersection of given ray (blockIdx.x) with n_faces faces (y block dimension).
  __device__ void cast_scene_intersection_step(grid_coord_t p, const grid_cell_t& cell, const ray_t& ray, const grid_t* grid, const face_t* faces)
  {
      extern __shared__ naive_cuda_ray_caster::distance_reduce_step_t distances[];

      // Current intersection distance for given ray and block's faces.
      naive_cuda_ray_caster::distance_reduce_step_t& step = distances[threadIdx.x];
      step.distance = FLT_MAX;
      step.face_idx = -1;

      int face_idx = threadIdx.x;

      // Intersection step.
      for (; face_idx < cell.count; face_idx += blockDim.x)
      {
          const int f = grid->triangles[cell.offset + face_idx];
          const face_t& triangle = faces[f];
          vec3 point;
          int check_result = cuda_math::triangle_intersect(ray.origin, ray.direction, triangle.points, &point);
          if (check_result == TRIANGLE_INTERSECTION_UNIQUE)
          {
              // Check if we hit geometry in the current cell.
              vec3 locator = (point - grid->base) / grid->side;
              if (p.x != (int)locator.x || p.y != (int)locator.y)
                  continue;

              vec3 space_distance = point - ray.origin;
              math::point_t new_distance = dot(space_distance, space_distance);
              if (new_distance < step.distance)
              {
                  step.distance = new_distance;
                  step.face_idx = f;
              }
          }
      }

      // Reduction cycle.
      __syncthreads();
  }

  /// @brief Reduces all distances for ray (blockIdx.x) and all faces.
  __device__ void cast_scene_reduction_step()
  {
      extern __shared__ naive_cuda_ray_caster::distance_reduce_step_t distances[];

      int thread_idx = threadIdx.x;
      int m = blockDim.x;

      while (m > 1)
      {
          int half_m = m >> 1;
          if (thread_idx < half_m)
          {
              if (distances[thread_idx].distance > distances[m - thread_idx - 1].distance)
              {
                  distances[thread_idx].distance = distances[m - thread_idx - 1].distance;
                  distances[thread_idx].face_idx = distances[m - thread_idx - 1].face_idx;
              }
          }
          __syncthreads();

          m -= half_m;
      }
  }

  __global__ void cast_cell(grid_coord_t p, callback_param_t param)
  {
      extern __shared__ naive_cuda_ray_caster::distance_reduce_step_t distances[];

      const grid_t* index = param.grid;
      const grid_cell_t& cell = index->cells[p.x + index->n_x * p.y];

      cast_scene_intersection_step(p, cell, param.ray, param.grid, param.faces);
      cast_scene_reduction_step();
      // Finalization step
      if (threadIdx.x == 0)
      {
          const ray_t& ray = param.ray;
          int face_idx = distances[0].face_idx;
          if (face_idx != -1)
          {
              int result_code = cuda_math::triangle_intersect(ray.origin, ray.direction, param.faces[face_idx].points, param.hit_point);
              *param.hit_face = result_code == TRIANGLE_INTERSECTION_UNIQUE ? distances[0].face_idx : -1;
          }
          else
          {
              *param.hit_face = -1;
          }
      }
  }
  
  __device__ bool traversal_handler(grid_coord_t p, callback_param_t* param)
  {
    const grid_t* index = param->grid;
    const grid_cell_t& cell = index->cells[p.x + index->n_x * p.y];

    if (!cell.count)
      return false;
    
    int concurrency = cell.count;
    if (concurrency > blockDim.x)
      concurrency = blockDim.x;
    int shared_size = concurrency * sizeof(naive_cuda_ray_caster::distance_reduce_step_t);

    cast_cell << <1, concurrency, shared_size >> >(p, *param);

    cudaDeviceSynchronize();

    // Don't continue traversal if we hit something.
    return *param->hit_face != -1;
  }

  __device__ void grid_traverse(const grid_t* grid, ray_t ray, callback_param_t* param)
  {
    if (!rebase_if_out_of_bound(grid, ray))
      return;

    vec3 u = ray.origin - grid->base;
    vec3 v = ray.direction - ray.origin;
    grid_coord_t p = { (int)(u.x / grid->side.x), (int)(u.y / grid->side.y) };

    // @todo Understand abs() logic - is it really required, or t can be negative.
    vec3 t_delta = fabs(grid->side / v); // 1 / velocity -> cells per v step
    vec3 base_distance = make_float3(v.x < 0 ? 0 : 1.f, v.y < 0 ? 0 : 1.f, 0);
    vec3 t = fabs(t_delta * (base_distance - fracf(u / grid->side))); // |(base_distance - fraction) is relative distance to next row.

    int step_x = v.x < 0 ? -1 : 1;
    int step_y = v.y < 0 ? -1 : 1;

    grid_coord_t stop = { v.x < 0 ? -1 : grid->n_x, v.y < 0 ? -1 : grid->n_y };

    assert(p.x >= 0 && p.x < grid->n_x);
    assert(p.y >= 0 && p.y < grid->n_y);

    while (true)
    {
      if (traversal_handler(p, param))
        return;

      const bool do_x = t.x <= t.y;
      const bool do_y = t.y <= t.x;

      if (do_x)
      {
        p.x += step_x;
        if (p.x == stop.x)
          return;
        t.x += t_delta.x;
      }

      if (do_y)
      {
        p.y += step_y;
        if (p.y == stop.y)
          return;
        t.y += t_delta.y;
      }
    }
  }

  __global__ void cast_scene(const face_t* faces, const grid_t* grid, const ray_t* rays, int n_rays, int* indices, vec3* points)
  {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_rays)
    {
      ray_t ray = rays[i];

      indices[i] = -1;
      callback_param_t param = { faces, grid, ray, &indices[i], &points[i] };
      grid_traverse(grid, ray, &param);
    }
  }

  /// @brief Creates virtual methods table from local methods.
  const ray_caster::system_methods_t methods =
  {
    (int(*)(ray_caster::system_t* system))&init,
    (int(*)(ray_caster::system_t* system))&shutdown,
    (int(*)(ray_caster::system_t* system, ray_caster::scene_t* scene))&set_scene,
    (int(*)(ray_caster::system_t* system))&prepare,
    (int(*)(ray_caster::system_t* system, ray_caster::task_t* task))&cast,
  };

  /// @brief Creates base system for ray caster.
  ray_caster::system_t* system_create()
  {
    cuda_system_t* s = (cuda_system_t*)malloc(sizeof(cuda_system_t));
    s->methods = &methods;
    return s;
  }
}

#define EPSILON   0.00000001

namespace cuda_math
{
  __device__ int triangle_intersect(vec3 origin, vec3 direction, const vec3* triangle, vec3* point)
  {
    vec3 u, v, n; // triangle vec3s
    vec3 dir, w0, w; // ray vec3s
    float r, a, b; // params to calc ray-plane intersect

    // get triangle edge vec3s and plane normal
    u = triangle[1] - triangle[0];
    v = triangle[2] - triangle[0];
    n = cross(u, v);              // cross product
    if (dot(n, n) < EPSILON)      // triangle is degenerate
      return -TRIANGLE_INTERSECTION_DEGENERATE; // do not deal with this case

    dir = direction - origin; // ray direction vec3
    w0 = origin - triangle[0];
    a = -dot(n, w0);
    b = dot(n, dir);
    if (fabs(b) < EPSILON) { // ray is  parallel to triangle plane
      if (a == 0)            // ray lies in triangle plane
        return -TRIANGLE_INTERSECTION_SAME_PLAIN;
      else return -TRIANGLE_INTERSECTION_DISJOINT; // ray disjoint from plane
    }

    // get intersect point of ray with triangle plane
    r = a / b;
    if (r < 0.0) // ray goes away from triangle
      return -TRIANGLE_INTERSECTION_DISJOINT; // => no intersect
    // for a segment, also test if (r > 1.0) => no intersect

    *point = origin + r * dir; // intersect point of ray and plane

    // is point inside triangle?
    float    uu, uv, vv, wu, wv, D;
    uu = dot(u, u);
    uv = dot(u, v);
    vv = dot(v, v);
    w = *point - triangle[0];
    wu = dot(w, u);
    wv = dot(w, v);
    D = uv * uv - uu * vv;

    // get and test parametric coords
    float s, t;
    s = (uv * wv - vv * wu) / D;
    if (s < 0.0 || s > 1.0)         // point is outside T
      return -TRIANGLE_INTERSECTION_DISJOINT;
    t = (uv * wu - uu * wv) / D;
    if (t < 0.0 || (s + t) > 1.0)  // point is outside T
      return -TRIANGLE_INTERSECTION_DISJOINT;

    return TRIANGLE_INTERSECTION_UNIQUE; // point is in T
  }
}
