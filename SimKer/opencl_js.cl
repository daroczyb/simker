__kernel void dotp(int N,int dim,int batch,__global float* ref,__global float* vec,__global float* out)
{
    int tidx = get_global_id(0);
    
    __private float sum_1=0.0;
    __private float sum_2=0.0;
    
    __private float m=0.0;
    __private float p=0.0;
    __private float q=0.0;
    
    __private int x = tidx/N;
    __private int y = tidx%N;
    
 //   out[y+x*N]=0.0;
    for(int d=0;d<dim;d++)
	{
	p=ref[y*dim+d];
	q=vec[d+dim*x];
	m=(p+q)/2;
        if(m>0 && p>0) sum_1+=p*log(p/m);
        if(m>0 && q>0) sum_2+=q*log(q/m);
	}
    out[y+x*N]=0.5*(sum_1+sum_2);
}

__kernel void dotp_tr(int N,int dim,__global float* ref,__global float* out)
{
    int tidx = get_global_id(0);
    
    __private float sum_1=0.0;
    __private float sum_2=0.0;
    
    __private float m=0.0;
    __private float p=0.0;
    __private float q=0.0;
    
    for(int r=tidx;r<N;r++)
    {
        out[tidx*N+r]=0.0;
        sum_1=0.0;
        sum_2=0.0;
        for(int d=0;d<dim;d++)
            {
            p=ref[tidx*dim+d];
	    q=ref[d+dim*r];
	    m=(p+q)/2;
    	    if(m>0 && p>0) sum_1+=p*log(p/m);
    	    if(m>0 && q>0) sum_2+=q*log(q/m);
	    }
        out[tidx*N+r]=0.5*(sum_1+sum_2);
    }
}
