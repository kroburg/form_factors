////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Template project which demonstrates the basics on how to setup a project
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <helper_math.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

#define _EPS 0.001f

typedef float point;
typedef float3 vec3;

inline __host__ __device__ vec3 make_vec3(point x, point y, point z)
{
  return make_float3(x, y, z);
}

typedef float4 vec4;

struct face_t
{
  vec3 points[3];
  vec3 bbox[2];
};

__global__ void face_init_bbox(face_t* f)
{
  f->bbox[0] = fminf(fminf(f->points[0], f->points[1]), f->points[2]);
  f->bbox[0] = fmaxf(fmaxf(f->points[0], f->points[1]), f->points[2]);
}

struct ray_t {
  vec3 origin;
  vec3 direction;
  vec3 inv_direction;
  int sign[3];
};

__global__ void ray_init(ray_t* ray, vec3 origin, vec3 direction) {
  vec3 inv_direction = 1.0f / direction;
  *ray = {
    origin,
    direction,
    inv_direction,
    { (inv_direction.x < 0) ? 1 : 0,
    (inv_direction.y < 0) ? 1 : 0,
    (inv_direction.z < 0) ? 1 : 0
    }
  };
}

__device__ float ray_box_intersection(const ray_t* r, const vec3* bbox) {
  vec3
    bMin = bbox[0],
    bMax = bbox[1];
  // x-slab intersections
  float t1 = (bMin.x - (r->origin.x))*(r->inv_direction.x);
  float t2 = (bMax.x - (r->origin.x))*(r->inv_direction.x);  
  float tnear = MIN(t1, t2);
  float tfar = MAX(t1, t2);

  // y-plane intersections
  t1 = (bMin.y-(r->origin.y))*(r->inv_direction.y);
  t2 = (bMax.y-(r->origin.y))*(r->inv_direction.y);
  tnear = MAX(tnear, MIN(t1, t2));
  tfar = MIN(tfar, MAX(t1, t2));

  // z-plane intersections
  t1 = (bMin.z-(r->origin.z))*(r->inv_direction.z);
  t2 = (bMax.z-(r->origin.z))*(r->inv_direction.z);
  tnear = MAX(tnear, MIN(t1, t2));
  tfar = MIN(tfar, MAX(t1, t2));
  return (tnear > tfar || tnear < _EPS ? -1.0f : tnear);
}

__device__ vec3 face_ray_intersection(const ray_t* r, const face_t* f) {
  vec3 tA = f->points[0];
  vec3 tB = f->points[1];
  vec3 tC = f->points[2];
  vec3 t1 = cross(tA - tC, r->direction);
  vec3 t2 = cross(tA - tB, tA - r->origin);
  float detA = dot(tA - tB, t1);
  /*float distance = dot(tC - tA, t2) / detA;
  if (distance <= minDistance || distance >= maxDistance)
    return make_vec3(-1.0f, -1.0f, -1.0f);
  float
    beta = ((tA - (r->origin))*temp_1) / detA;
  if (beta < 0.0f)
    return make_vec3(-1.0f, -1.0f, -1.0f);
  float
    gamma = ((r->direction)*temp_2) / detA;
  if (0.0f <= gamma && (beta + gamma) <= 1.0f)
    return make_vec3(distance, beta, gamma);*/
  return make_vec3(-1.0f, -1.0f, -1.0f);
}

__global__ void setup_kernel(curandState *state)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(1234, id, 0, &state[id]);
}


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
//void runTest(int argc, char **argv);
//


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
//int
//main(int argc, char **argv)
//{
//    runTest(argc, argv);
//}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char **argv)
{
    bool bTestResult = true;

    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);


    dim3 threadsPerBlock(64);
    dim3 numBlocks(64);
    curandState *randomStates;
    checkCudaErrors(cudaMalloc((void **)&randomStates, numBlocks.x * threadsPerBlock.x * sizeof(curandState)));
    setup_kernel<<<numBlocks, threadsPerBlock>>>(randomStates);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits   
    cudaDeviceReset();
    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
