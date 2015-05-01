#include "system.cuh"
#include "system.h"

#include <helper_cuda.h>
#include <helper_math.h>

#include <assert.h>

using ray_caster::system_t;
namespace cuda_ray_caster
{
  struct cuda_system_t : system_t
  {
    int n_faces;
    face_t* faces;
    int dev_id;
    int n_tpb;
  };

  int init(cuda_system_t* system)
  {
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
    cudaDeviceReset();
    return RAY_CASTER_OK;
  }

  int set_scene(cuda_system_t* system, scene_t* scene)
  {
    if (system->faces)
    {
      cudaFree(system->faces);
      system->faces = 0;
      system->n_faces = 0;
    }

    system->n_faces = scene->n_faces;

    if (system->n_faces == 0)
      return RAY_CASTER_OK;

    if (system->n_faces > 2048)
      return -RAY_CASTER_OUT_OF_RANGE;

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

  const ray_caster::system_methods_t methods =
  {
    (int(*)(system_t* system))&init,
    (int(*)(system_t* system))&shutdown,
    (int(*)(system_t* system, ray_caster::scene_t* scene))&set_scene,
    (int(*)(system_t* system))&prepare
  };

  system_t* system_create()
  {
    cuda_system_t* s = (cuda_system_t*)malloc(sizeof(cuda_system_t));
    s->methods = &methods;
    return s;
  }

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
}
