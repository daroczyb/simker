//
//  OpenCL_env.cpp
//  SimKer
//
//  Created by Daroczy Balint on 07/01/2015.
//  Copyright (c) 2015 Daroczy Balint. All rights reserved.
//
#include <string.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <time.h>
#include "OpenCL_env.h"

#define MAXKP 1048576

using namespace std;

OpenCL_env::OpenCL_env(int type)
{
    set=0;
    count=0;
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if(type==2) device=CL_DEVICE_TYPE_GPU;
    else device=CL_DEVICE_TYPE_CPU;
    ret = clGetDeviceIDs( platform_id, device, 1,
                         &device_id, &ret_num_devices);
    
    if(ret_num_devices==0)
    {
        if(device==CL_DEVICE_TYPE_CPU) printf("No OpenCL compatible CPU!\n");
        else printf("No OpenCL compatible GPU!\n");
        return;
    }
    else set=1;
    dbg=0;
    sprintf(ext,"");
    context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    if(ret != CL_SUCCESS) {printf ("inti failed\n"); set=0; return;}
    else {printf("Init done\n");}
}

OpenCL_env::OpenCL_env(char* fname,int type)
{
    set=0;
    count=0;
    desc_type=1;
    if(type==2) device=CL_DEVICE_TYPE_GPU;
    else device=CL_DEVICE_TYPE_CPU;
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id,device,1,&device_id, &ret_num_devices);
    
    printf("Try to connect device: %d\n",device);
    
    if(ret_num_devices==0)
    {
        if(device==CL_DEVICE_TYPE_CPU) printf("No OpenCL compatible CPU!\n");
        else printf("No OpenCL compatible GPU!\n");
        return;
    }
    else set=1;
    
    dbg=0;
    
    if(device==CL_DEVICE_TYPE_CPU)
    {
        global_item_size=1024;
        local_item_size=1;
    }
    else
    {
        size_t vals[3];
        ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(vals), (void*)vals, NULL);
        printf("Max work item sizes: %d,%d,%d\n",(int)vals[0],(int)vals[1],(int)vals[2]);
        global_item_size=(int)vals[0];
        local_item_size=(int)vals[2];
        if(local_item_size>=global_item_size/2) local_item_size/=4;
    }
    if(device==CL_DEVICE_TYPE_GPU) printf("Number of gpus: %d -> workers: %d (%d) \n",ret_num_devices,(int)global_item_size,(int)local_item_size);
    else printf("Number of cpus: %d -> workers: %d (%d)\n",ret_num_devices,(int)global_item_size,(int)local_item_size);
    GetInfo();
    
    context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    if(ret != CL_SUCCESS) {printf ("init failed\n"); set=0; return;}
    SetSource(fname);
}


OpenCL_env::~OpenCL_env()
{
    if(set)
    {
        
        if(set_ref)
        {
            delete[] ref;
            ret = clReleaseMemObject(ref_d);
            ret = clReleaseMemObject(vec_d);
        }
        ret = clFlush(command_queue);
        if(ret!=CL_SUCCESS) printf("clFLush fail\n");
        ret = clFinish(command_queue);
        if(ret!=CL_SUCCESS) printf("clFinish fail\n");
        //    while(CL_SUCCESS!=clFinish(command_queue)) {}
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);
        ret = clReleaseProgram(program);
        printf("CL finished\n");
    }
}

void OpenCL_env::GetInfo()
{
    char buffer[500];
    ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(buffer), (void*)buffer, NULL);
    printf("Device name: %s\n",buffer);
    ret = clGetDeviceInfo(device_id, CL_DEVICE_PROFILE, sizeof(buffer), (void*)buffer, NULL);
    printf("Profile: %s\n",buffer);
    size_t vals[3];
    ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(vals), (void*)vals, NULL);
    printf("Max work item sizes: %d,%d,%d\n",(int)vals[0],(int)vals[1],(int)vals[2]);
}

void OpenCL_env::SetSource(char* fname)
{
    FILE* programHandle;
    char *programBuffer;
    size_t programSize;
    
    programHandle = fopen(fname, "r");
    fseek(programHandle, 0, SEEK_END);
    programSize = ftell(programHandle);
    rewind(programHandle);
    programBuffer = (char*) malloc(programSize + 1);
    programBuffer[programSize] = '\0';
    fread(programBuffer, sizeof(char), programSize, programHandle);
    fclose(programHandle);
    program = clCreateProgramWithSource(context, 1,(const char**) &programBuffer, &programSize, NULL);
    free(programBuffer);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if(ret!=CL_SUCCESS)
    {
        printf("progam failed\n"); set=0;
        if(ret==CL_INVALID_PROGRAM) printf ("CL_INVALID_PROGRAM\n");
        if(ret==CL_INVALID_VALUE) printf ("CL_INVALID_VALUE\n");
        if(ret==CL_INVALID_VALUE) printf ("CL_INVALID_VALUE\n");
        if(ret==CL_INVALID_DEVICE) printf ("CL_INVALID_DEVICE\n");
        if(ret==CL_INVALID_BINARY) printf ("CL_INVALID_BINARY\n");
        if(ret==CL_INVALID_BUILD_OPTIONS) printf ("CL_INVALID_BUILD_OPTIONS\n");
        if(ret==CL_INVALID_OPERATION) printf ("CL_INVALID_OPERATION\n");
        if(ret==CL_COMPILER_NOT_AVAILABLE) printf ("CL_COMPILER_NOT_AVAILABLE\n");
        if(ret==CL_BUILD_PROGRAM_FAILURE) printf ("CL_BUILD_PROGRAM_FAILURE\n");
        if(ret==CL_INVALID_OPERATION) printf ("CL_INVALID_OPERATION\n");
        if(ret==CL_OUT_OF_HOST_MEMORY) printf ("CL_OUT_OF_HOST_MEMORY\n");
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *) malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("%s\n", log);
    }
    else printf("Source ok\n");
}

void OpenCL_env::SetRef(char* reffile,int dim1,int sub_dim,int str,int end,int num1,int batch)
{
    FILE* f=fopen(reffile,"r");
    if(f==NULL) {set_ref=0; return;}
    fseek(f,0,SEEK_SET);
    
    char label[500];
    //    fscanf(f,"%s %d %s %d",label,&ref_num,label,&ref_dim);
    ref_num=num1;
    ref_dim=end-str;
    
    if(str==0) ref=new float[ref_dim*ref_num];
    
    float val;
    
    for(int i=0;i<ref_num;i++)
    {
        fscanf(f,"%s",label);
        //	printf("label: %s\n",label);
        string imgname;
        if(str==0)
        {
            imgname.append(label);
            tr_name.push_back(imgname);
        }
        for(int d=0;d<dim1;d++)
        {
            fscanf(f,"%f",&val);
            if(d>=str && d<end)
                ref[i*ref_dim+d-str]=val;
        }
    }
    fclose(f);
    
    if(str==0)
    {
        ref_d = clCreateBuffer(context,CL_MEM_READ_ONLY,ref_dim*ref_num*sizeof(float),NULL,&ret);
        vec_d = clCreateBuffer(context,CL_MEM_READ_ONLY,batch*ref_dim*sizeof(float),NULL,&ret);
    }
    ret = clEnqueueWriteBuffer(command_queue,ref_d,CL_TRUE,0,ref_dim*ref_num*sizeof(float),ref,0,NULL,NULL);
    
}

void OpenCL_env::DotpCL(float* vec,int batch,int part)
{
    kernel = clCreateKernel(program, "dotp", &ret);
    
    global_item_size=ref_num*batch;
    
    if(kernel==NULL)
    {
        printf("	dotp CL kernel: error!\n");
        return;
    }
    //    else printf("	dotp CL kernel: ok! (%d)\n",(int)global_item_size);
    
    float* out=new float[ref_num*batch];
    cl_mem out_d = clCreateBuffer(context,CL_MEM_WRITE_ONLY,batch*ref_num*sizeof(float),NULL,&ret);
    ret = clEnqueueWriteBuffer(command_queue,vec_d,CL_TRUE,0,batch*ref_dim*sizeof(float),vec,0,NULL,NULL);
    
    ret = clSetKernelArg(kernel,0,sizeof(int),(void *)&ref_num);
    ret = clSetKernelArg(kernel,1,sizeof(int),(void *)&ref_dim);
    ret = clSetKernelArg(kernel,2,sizeof(int),(void *)&batch);
    ret = clSetKernelArg(kernel,3,sizeof(cl_mem),(void *)&ref_d);
    ret = clSetKernelArg(kernel,4,sizeof(cl_mem),(void *)&vec_d);
    ret = clSetKernelArg(kernel,5,sizeof(cl_mem),(void *)&out_d);
    
    ret = clEnqueueNDRangeKernel(command_queue,kernel,1,NULL,&global_item_size,NULL,0,NULL,NULL);
    
    ret = clEnqueueReadBuffer(command_queue,out_d,CL_TRUE,0,batch*ref_num*sizeof(float),out,0,NULL,NULL);
    
    char outfilename[500];
    sprintf(outfilename,"%s.te.ker.p%d",outfile,part);
    FILE* f=fopen(outfilename,"a");

    for(int j=0;j<batch;j++)
        {
        fprintf(f,"%s",te_name[j].c_str());
        for(int i=0;i<ref_num;i++)
            fprintf(f," %g",out[j*ref_num+i]);
        fprintf(f,"\n");
        }
    fclose(f);
    delete[] out;
    ret = clReleaseMemObject(out_d);
}
void OpenCL_env::DotpCL_tr(int part)
{
    kernel = clCreateKernel(program, "dotp_tr", &ret);
    
    global_item_size=ref_num;
    
    if(kernel==NULL)
    {
        printf("	dotp train CL kernel: error!\n");
        return;
    }
    else printf("	dotp train CL kernel: ok! (%d)\n",(int)global_item_size);
    
    float* out=new float[ref_num*ref_num];
    cl_mem out_d = clCreateBuffer(context,CL_MEM_WRITE_ONLY,ref_num*ref_num*sizeof(float),NULL,&ret);
    
    ret = clSetKernelArg(kernel,0,sizeof(int),(void *)&ref_num);
    ret = clSetKernelArg(kernel,1,sizeof(int),(void *)&ref_dim);
    ret = clSetKernelArg(kernel,2,sizeof(cl_mem),(void *)&ref_d);
    ret = clSetKernelArg(kernel,3,sizeof(cl_mem),(void *)&out_d);
    
    ret = clEnqueueNDRangeKernel(command_queue,kernel,1,NULL,&global_item_size,NULL,0,NULL,NULL);
    
    ret = clEnqueueReadBuffer(command_queue,out_d,CL_TRUE,0,ref_num*ref_num*sizeof(float),out,0,NULL,NULL); 
    
    char outfilename[500];
    sprintf(outfilename,"%s.tr.ker.p%d",outfile,part);
    FILE* f=fopen(outfilename,"a");
    for(int i=0;i<ref_num;i++)
    {
        fprintf(f,"%s",tr_name[i].c_str());
        for(int j=0;j<i;j++)
            fprintf(f," %g",out[j*ref_num+i]);
        for(int j=i;j<ref_num;j++)
            fprintf(f," %g",out[i*ref_num+j]);
        fprintf(f,"\n");
    }
    fclose(f);
    delete[] out;
    ret = clReleaseMemObject(out_d);
}
