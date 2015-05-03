#include "system.cuh"
#include "cuda_system.h"

#include <helper_cuda.h>
#include <helper_math.h>
#include <bulk/bulk.hpp>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include <assert.h>

using ray_caster::system_t;
namespace cuda_ray_caster
{
  struct cuda_system_t : system_t
  {
    ray_caster::scene_t* scene;
    int n_faces;
    face_t* faces;
    int dev_id;
    int n_tpb;
  };

  int init(cuda_system_t* system)
  {
    system->scene = 0;
    system->n_faces = 0;
    system->faces = 0;
    const char* argv[] = { "" };
    system->dev_id = findCudaDevice(1, (const char **)argv);
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, system->dev_id));
    system->n_tpb = deviceProp.maxThreadsPerBlock;

    return RAY_CASTER_OK;
  }

  int shutdown(cuda_system_t* system)
  {
    if (system->faces)
      cudaFree(system->faces);
    system->scene = 0;
    cudaDeviceReset();
    return RAY_CASTER_OK;
  }

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
      return RAY_CASTER_OK;

    ray_caster::face_t* sourceFaces;
    checkCudaErrors(cudaMalloc((void**)&sourceFaces, scene->n_faces * sizeof(ray_caster::face_t)));
    checkCudaErrors(cudaMemcpy(sourceFaces, scene->faces, sizeof(ray_caster::face_t) * scene->n_faces, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&system->faces, scene->n_faces * sizeof(face_t)));
    
    // @todo How to test that faces upload working?
    // @todo Decide on "perfect occupancy". WTF?
    int n_tpb = system->n_tpb;
    int n_blocks = (system->n_faces + n_tpb - 1) / n_tpb;
    load_scene_faces << <n_blocks, n_tpb >> >(sourceFaces, system->faces, system->n_faces);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(sourceFaces));

    return RAY_CASTER_OK;
  }

  int prepare(cuda_system_t* system)
  {
    if (system->n_faces == 0)
      return -RAY_CASTER_ERROR;
    return RAY_CASTER_OK;
  }

  vec3 ext2loc(ray_caster::vec3 a)
  {
    return make_float3(a.x, a.y, a.z);
  }

  ray_caster::vec3 loc2ext(vec3 a)
  {
    return ray_caster::make_vec3(a.x, a.y, a.z);
  }

  struct client_cast_result_t
  {
    int face_index;
    vec3 point;
  };

  struct bulk_find_min
  {
    __host__ __device__
    void operator()(bulk::agent<> &self, float* distances, const vec3* points, int n_faces, client_cast_result_t* find_results)
    {
      int i = self.index();
      typedef thrust::device_ptr<float> distance_ptr;
      distance_ptr thrust_results(distances + i * n_faces);
      distance_ptr result_with_min_distance = thrust::min_element(thrust::device, thrust_results, thrust_results + n_faces);
      
      if (*result_with_min_distance != FLT_MAX)
      {
        int face_index = result_with_min_distance - thrust_results;
        find_results[i].face_index = face_index;
        find_results[i].point = points[i * n_faces + face_index];
      }
      else
      {
        find_results[i].face_index = -1;
      }
    }
  };

  int cast(cuda_system_t* system, ray_caster::task_t* task)
  {
    using namespace ray_caster;
    
    const int n_faces = system->n_faces;
    const int max_concurrent_rays = 8192;
    
    thrust::device_vector<float> distances(n_faces * max_concurrent_rays);
    thrust::device_vector<vec3> points(n_faces * max_concurrent_rays);
    thrust::device_vector<client_cast_result_t> d_find_results(max_concurrent_rays);
    thrust::host_vector<client_cast_result_t> h_find_results;

    ray_t* d_rays;
    checkCudaErrors(cudaMalloc((void**)&d_rays, task->n_tasks * sizeof(ray_t)));
    checkCudaErrors(cudaMemcpy(d_rays, task->ray, task->n_tasks * sizeof(ray_t), cudaMemcpyHostToDevice));

    for (int t = 0; t < task->n_tasks;)
    {
      int n_tpb = system->n_tpb < system->n_faces ? system->n_tpb : system->n_faces;
      int n_blocks = (system->n_faces + n_tpb - 1) / n_tpb;
      int n_rays = t + max_concurrent_rays < task->n_tasks ? max_concurrent_rays : task->n_tasks - t;
      dim3 grid(n_blocks, n_rays);
      cast_scene_faces << <grid, n_tpb >> >(system->faces, system->n_faces, d_rays + t, thrust::raw_pointer_cast(distances.data()), thrust::raw_pointer_cast(points.data()));
      checkCudaErrors(cudaPeekAtLastError());
      checkCudaErrors(cudaDeviceSynchronize());

      bulk::async(bulk::par(n_rays), bulk_find_min(), bulk::root.this_exec, thrust::raw_pointer_cast(distances.data()), thrust::raw_pointer_cast(points.data()), n_faces, thrust::raw_pointer_cast(d_find_results.data()));
      h_find_results = d_find_results;

      for (int j = 0; j != n_rays; ++j)
      {
        if (h_find_results[j].face_index != -1)
        {
          task->hit_face[t + j] = system->scene->faces + h_find_results[j].face_index;
          task->hit_point[t + j] = loc2ext(h_find_results[j].point);
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

  const ray_caster::system_methods_t methods =
  {
    (int(*)(system_t* system))&init,
    (int(*)(system_t* system))&shutdown,
    (int(*)(system_t* system, ray_caster::scene_t* scene))&set_scene,
    (int(*)(system_t* system))&prepare,
    (int(*)(ray_caster::system_t* system, ray_caster::task_t* task))&cast,
  };

  system_t* system_create()
  {
    cuda_system_t* s = (cuda_system_t*)malloc(sizeof(cuda_system_t));
    s->methods = &methods;
    return s;
  }

#define EPSILON   0.00000001

  __device__ void init_face_bbox(face_t* f)
  {
    f->bbox[0] = fminf(fminf(f->points[0], f->points[1]), f->points[2]);
    f->bbox[0] = fmaxf(fmaxf(f->points[0], f->points[1]), f->points[2]);
  }

  __global__ void load_scene_faces(const ray_caster::face_t* source, face_t* target, int n_faces)
  {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n_faces)
    {
      // @todo Implement some kind of assertion here.
      //assert(sizeof(face_t::points) == sizeof(ray_caster::face_t::points));
      memcpy(target[i].points, source[i].points, sizeof(face_t::points));
      init_face_bbox(&target[i]);
    }
  }

  __global__ void cast_scene_faces(const face_t* faces, int n_faces, const ray_t* rays, float* distances, vec3* points)
  {
    int face_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int ray_idx = blockIdx.y;
    int result_idx = n_faces * blockIdx.y + face_idx;

    if (face_idx < n_faces)
    {
      ray_t ray = rays[ray_idx];
      int result_code = triangle_intersect(ray, faces[face_idx].points, &points[result_idx]);
      if (result_code == TRIANGLE_INTERSECTION_UNIQUE)
      {
        vec3 distance = points[result_idx] - ray.origin;
        distances[result_idx] = dot(distance, distance);
      }
      else
      {
        distances[result_idx] = FLT_MAX;
      }
    }
  }

  __device__ int triangle_intersect(ray_t ray, const vec3* triangle, vec3* point)
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

    dir = ray.direction - ray.origin; // ray direction vec3
    w0 = ray.origin - triangle[0];
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

    *point = ray.origin + r * dir; // intersect point of ray and plane

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
