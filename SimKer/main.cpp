//
//  main.cpp
//  SimKer
//
//  Created by Daroczy Balint on 07/01/2015.
//  Copyright (c) 2015 Daroczy Balint. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <string.h>
#include <vector>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifdef __APPLE__
#include "OpenCL_env.h"
#else
#include <OpenCL/OpenCL_env.h>
#endif

using namespace std;

void usage()
{
    printf("GPU based Similarity kernel, created by Balint Daroczy on 07/01/2015\n");
    printf("\nUsage: <approximaton set> <options>\n");
    printf("\nOptions:\n");
    printf("        -kernel : OpenCL kernel file\n");
    printf("        -GPU : use GPU instead of CPU\n");
    printf("        -split: split the vectors into parts\n");
    printf("        -batch: split the reference set into batches\n");
    printf("        -ntr : number of reference points\n");
    printf("        -nte : number of test instances\n");
    printf("        -sim : dimension of instance vectors\n");
    printf("        -output : output filename (<>.tr.ker.p* and <>.te.ker.p*)\n");
    printf("        -test : test file \n");
    printf("        -test_only : skip reference set\n");
}

int main(int argc,char* argv[])
{
    unsigned long int act_clock=clock();
    unsigned long int str_clock=act_clock;
    
    char kernel_file[500];
    char train_file[500];
    char test_file[500];
    char output_file[500];
    
    int device=1;
    int ntr=1;
    int nte=1;
    int dim=1;
    int split=1;
    int batch=1;
    
    int train_sim=1;
    int test=0;
    
#ifdef __APPLE__
     sprintf(train_file,"/Users/balint/Desktop/SimKer/ref.txt");
     sprintf(test_file,"/Users/balint/Desktop/SimKer/test.txt");
     sprintf(output_file,"/Users/balint/Desktop/SimKer/test.out");
     sprintf(kernel_file,"/Users/balint/Desktop/SimKer/SimKer/SimKer/opencl_sim.cl");
     ntr=10;
     dim=3;
     nte=10;
     test=1;
     batch=2;
#else
    if(argc<2) {usage(); return 1;}
    sprintf(kernel_file,"OpenCL/opencl_sim.cl");
#endif
    
    if(argc>1)
        {
        sprintf(train_file,"%s",argv[1]);
        }

    printf("Approximation set: %s\n",train_file);
    
    for(int i=2;i<argc;i++)
    {
        if(strcmp(argv[i],"-ntr")==0) {ntr=atoi(argv[i+1]);}
        if(strcmp(argv[i],"-nte")==0) {nte=atoi(argv[i+1]);}
        if(strcmp(argv[i],"-dim")==0) {dim=atoi(argv[i+1]); printf("Dimension: %d\n",dim);}
        if(strcmp(argv[i],"-split")==0) {split=atoi(argv[i+1]); printf("Split into %d pieces (sub_dim: %d)\n",split,dim/split);}
        if(strcmp(argv[i],"-output")==0) {sprintf(output_file,"%s",argv[i+1]); printf("Output file: %s.tr.ker.p* and %s.te.ker.p*\n",output_file,output_file);}
        if(strcmp(argv[i],"-GPU")==0) {device=2; printf("Compute device: GPU\n");}
        if(strcmp(argv[i],"-test_only")==0) {train_sim=0; printf("Test only\n");}
        if(strcmp(argv[i],"-test")==0) {test=1; sprintf(test_file,"%s",argv[i+1]); printf("Test file: %s\n",test_file);}
        if(strcmp(argv[i],"-batch")==0) {batch=atoi(argv[i+1]); printf("Batch size: %d\n",batch);}
        if(strcmp(argv[i],"-kernel_file")==0) {    sprintf(kernel_file,"%s",argv[i+1]); printf("Kernel file: %s\n",kernel_file);}
    }
    
    OpenCL_env opencl(kernel_file,device);
    
    FILE* f=fopen(test_file,"r");
    
    int sub_dim=dim/split;
    printf("sub_dim: %d\n",sub_dim);
    float* vec=new float[sub_dim*batch];
    
    for(int part=0;part<split;part++)
    {
        int str=sub_dim*part;
        int end=(part+1)*sub_dim;
        printf("part %d -> %d..%d\n",part,str,end);
        
        act_clock = clock();
        
        opencl.SetRef(train_file,dim,sub_dim,str,end,ntr,batch);
      
        printf("Loading was done in %.4f sec (%dx%d)\n",(float)(clock() - act_clock)/CLOCKS_PER_SEC,ntr,dim);
        
        sprintf(opencl.outfile,"%s",output_file);
        
        if(train_sim==1)
            {
            act_clock=clock();
            opencl.DotpCL_tr(part);
            printf("Train set similarity done: %.4f sec\n",(float)(clock() - act_clock)/CLOCKS_PER_SEC);
            }
        float val;
        act_clock=clock();
        if(test==1)
            {
            act_clock=clock();
            fseek(f,0,SEEK_SET);
            int act=0;
            int act_batch=0;
            char label[500];
            while(act<nte)
                {
                for(int i=0;i<batch;i++)
                    {
                        if(act<nte)
                            {
                                fscanf(f,"%s",label);
                                string imgname;
                                if(str==0)
                                {
                                    imgname.append(label);
                                    opencl.te_name.push_back(imgname);
                                }
                                for(int d=0;d<dim;d++)
                                    {
                                    fscanf(f,"%f",&val);
                                    if(d>=str && d<end)
                                        {
                                        vec[i*sub_dim+(d-str)]=val;
                                        }
                                    }
                            act_batch++;
                            }
                        act++;
                    }
                printf("%dth: %s (%d..%d batch size: %d)\n",act,opencl.te_name[act_batch-1].c_str(),str,end-1,act_batch);
                opencl.DotpCL(vec,act_batch,part);
                act_batch=0;
                opencl.te_name.clear();
                }
            printf("Test set: %.4f sec\n",(float)(clock() - act_clock)/CLOCKS_PER_SEC);
            }
    }
    
    if(f!=NULL) fclose(f);
    
    delete[] vec;
    
    printf("Running time: %.4f sec\n",(float)(clock() - str_clock)/CLOCKS_PER_SEC);
    
    return 0;
}
