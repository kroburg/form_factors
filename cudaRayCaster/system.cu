#include "system.cuh"
#include "cuda_system.h"

#include <helper_cuda.h>
#include <helper_math.h>
#include <thrust/extrema.h>
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

  struct ray_cast_task_min_distance
  {
    __host__ __device__
    bool operator()(const cast_result_t& lhs, const cast_result_t& rhs)
    {
      return lhs.distance < rhs.distance;
    }
  };

  int cast(cuda_system_t* system, ray_caster::task_t* task)
  {
    using namespace ray_caster;

    const int results_mem_size = system->n_faces * sizeof(cast_result_t);
    cast_result_t* h_results = (cast_result_t*)malloc(results_mem_size);
    cast_result_t* d_results;
    checkCudaErrors(cudaMalloc((void**)&d_results, results_mem_size));

    for (int t = 0; t != task->n_tasks; ++t)
    {
      ray_caster::ray_t ray = task->ray[t];

      checkCudaErrors(cudaMemset(d_results, 0, results_mem_size));

      int n_tpb = system->n_tpb;
      int n_blocks = (system->n_faces + n_tpb - 1) / n_tpb;
      cast_scene_faces << <n_blocks, n_tpb >> >(system->faces, system->n_faces, ext2loc(ray.origin), ext2loc(ray.direction), d_results);
      checkCudaErrors(cudaPeekAtLastError());
      checkCudaErrors(cudaDeviceSynchronize());

      typedef thrust::device_ptr<cast_result_t> cast_result_ptr;
      cast_result_ptr thrust_results(d_results);
      cast_result_ptr result_with_min_distance = thrust::min_element(thrust_results, thrust_results + system->n_faces, ray_cast_task_min_distance());
      checkCudaErrors(cudaDeviceSynchronize());
      int face_id = result_with_min_distance - thrust_results;
      cast_result_t result;
      checkCudaErrors(cudaMemcpy(&result, result_with_min_distance.get(), sizeof(cast_result_t), cudaMemcpyDeviceToHost));
      task->hit_face[t] = system->scene->faces + face_id;
      task->hit_point[t] = loc2ext(result.point);
    }

    checkCudaErrors(cudaFree(d_results));
    free(h_results);
      
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

  __global__ void cast_scene_faces(const face_t* faces, int n_faces, vec3 origin, vec3 direction, cast_result_t* results)
  {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n_faces)
    {
      ray_t ray = { origin, direction };
      cast_result_t& result = results[i];
      result.face = &faces[i];
      result.result_code = triangle_intersect(ray, faces[i].points, &result.point);
      if (result.result_code == TRIANGLE_INTERSECTION_UNIQUE)
      {
        vec3 distance = results[i].point - origin;
        result.distance = dot(distance, distance);
      }
      else
      {
        result.distance = FLT_MAX;
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
