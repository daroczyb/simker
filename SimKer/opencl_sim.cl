__kernel void dotp(int N,int dim,int batch,__global float* ref,__global float* vec,__global float* out)
{
    int tidx = get_global_id(0);
    
    __private float sum=0.0;
    
    __private int x = tidx/N;
    __private int y = tidx%N;
    
    for(int d=0;d<dim;d++)
        sum+=ref[y*dim+d]*vec[d+dim*x];
    out[y+x*N]=sum;
}

__kernel void dotp_tr(int N,int dim,__global float* ref,__global float* out)
{
    int tidx = get_global_id(0);
    
    __private float sum=0.0;
    
    for(int r=tidx;r<N;r++)
    {
        out[tidx*N+r]=0.0;
        sum=0.0;
        for(int d=0;d<dim;d++)
            sum+=ref[tidx*dim+d]*ref[r*dim+d];
        out[tidx*N+r]=sum;
    }
}
