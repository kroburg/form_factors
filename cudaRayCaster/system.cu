#include "system.cuh"
#include "cuda_system.h"

#include <helper_cuda.h>
#include <helper_math.h>
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
    //system->n_tpb = deviceProp.maxThreadsPerBlock;
    system->n_tpb = 128;

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

  vec3 ext2loc(math::vec3 a)
  {
    return make_float3(a.x, a.y, a.z);
  }

  math::vec3 loc2ext(vec3 a)
  {
    return math::make_vec3(a.x, a.y, a.z);
  }

  struct distance_reduce_step_t
  {
    float distance;
    int face_idx;
  };

  int cast(cuda_system_t* system, ray_caster::task_t* task)
  {
    using namespace ray_caster;

    const int n_rays = task->n_tasks;

    const int max_concurrent_rays = (int)pow(2, ceil(log2(16384 * 1024 / (system->n_faces + 1023))));

    thrust::device_vector<vec3> points(max_concurrent_rays);
    thrust::device_vector<int> indices(max_concurrent_rays);
    thrust::host_vector<vec3> h_points;
    thrust::host_vector<int> h_indices;

    ray_t* d_rays;
    checkCudaErrors(cudaMalloc((void**)&d_rays, n_rays * sizeof(ray_t)));
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
    f->bbox[1] = fmaxf(fmaxf(f->points[0], f->points[1]), f->points[2]);
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

  __global__ void load_rays(const math::ray_t* source, ray_t* target, int n_rays)
  {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n_rays)
    {
      memcpy(&target[i], &source[i], 2 * sizeof(vec3));
      target[i].inv_dir = make_float3(1.) / (target[i].direction - target[i].origin);
    }
  }

  __device__ void cast_scene_intersection_step(const face_t* faces, int n_faces, const ray_t* rays, int* indices, vec3* points)
  {
    extern __shared__ distance_reduce_step_t distances[];

    distance_reduce_step_t& step = distances[threadIdx.y];
    step.distance = FLT_MAX;
    step.face_idx = -1;

    int face_idx = threadIdx.y;
    const int ray_idx = blockIdx.x;

    // Intersection step.
    for (; face_idx < n_faces; face_idx += blockDim.y)
    { 
      ray_t ray = rays[ray_idx];
      if (face_bbox_intersect(ray, &faces[face_idx]))
      {
        vec3 point;
        int result_code = triangle_intersect(ray, faces[face_idx].points, &point);
        if (result_code == TRIANGLE_INTERSECTION_UNIQUE)
        {
          point -= ray.origin;
          float distance = dot(point, point);
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

  __device__ void cast_scene_reduction_step(const face_t* faces, int n_faces, const ray_t* rays, int* indices, vec3* points)
  { 
    extern __shared__ distance_reduce_step_t distances[];

    int thread_idx = threadIdx.y;
    int n_blocks = (n_faces + blockDim.y - 1) / blockDim.y;
    int m = blockIdx.y == n_blocks ? n_faces % blockDim.y : blockDim.y;
    
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

  __global__ void cast_scene_faces_with_reduction(const face_t* faces, int n_faces, const ray_t* rays, int* indices, vec3* points)
  {
    extern __shared__ distance_reduce_step_t distances[];

    int thread_idx = threadIdx.y;
    int ray_idx = blockIdx.x;

    cast_scene_intersection_step(faces, n_faces, rays, indices, points);
    cast_scene_reduction_step(faces, n_faces, rays, indices, points);
    // Finalization step
    if (thread_idx == 0)
    {
      ray_t ray = rays[ray_idx];
      int face_idx = distances[0].face_idx;
      int result_code = triangle_intersect(ray, faces[face_idx].points, &points[ray_idx]);
      indices[ray_idx] = result_code == TRIANGLE_INTERSECTION_UNIQUE ? distances[0].face_idx : -1;
    }
  }

  __device__ bool face_bbox_intersect(ray_t ray, const face_t* face)
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
