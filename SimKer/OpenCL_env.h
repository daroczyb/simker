//
//  OpenCL_env.h
//  SimKer
//
//  Created by Daroczy Balint on 07/01/2015.
//  Copyright (c) 2015 Daroczy Balint. All rights reserved.
//

#ifndef __SimKer__OpenCL_env__
#define __SimKer__OpenCL_env__

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

class OpenCL_env
{
public:
    OpenCL_env(int type);
    OpenCL_env(char* fname,int type);
    ~OpenCL_env();
    void SetSource(char* fname);
    void GetInfo();
    
    void SetRef(char* reffile,int dim,int sub_dim,int str,int end,int num,int batch);

    void DotpCL(float* vec,int batch,int part);
    void DotpCL_tr(int part);
    
    int device;
    int set;
    int set_ref;
    int count;
    int dbg;
    int desc_type;
    
    int K;
    int N;
    int ref_num;
    int ref_dim;
    
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_context context;
    cl_int ret;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
    size_t global_item_size;
    size_t local_item_size;
    
    char outfile[500];
    char ext[500];
    
    float* res;
    float* ref;
    
    cl_mem vec_d;
    cl_mem ref_d;
    
    std::vector <std::string> tr_name;
    std::vector <std::string> te_name;
};

#endif /* defined(__SimKer__OpenCL_env__) */
