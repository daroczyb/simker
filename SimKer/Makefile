#OPT= -march=core2 -mtune=generic -pipe -O2
OPT= -O3

#OPENCV_LINUX = `pkg-config opencv --cflags --libs` 
#OPENCV_LINUX = -I/usr/include/opencv -I/usr/include/opencv2 -L/usr/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -I./
#OPENCV_LINUX = -I/usr/include/opencv -I. -L/usr/lib -lopencv_core -lopencv_highgui -lopencv_imgproc
#OPENCV_MAC =  -I/opt/local/include/opencv -L/opt/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -I./
#OPENCL = /home/daroczyb/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc/ -lOpenCL
#OPENCL_INC= /home/daroczyb/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc/
OPENCL = /opt/AMDAPP/lib/x86_64/ -l OpenCL
OPENCL_INC= /opt/AMDAPP/include/
OPENCL_MAC= -framework OpenCL
SDKINC = /home/daroczyb/NVIDIA_GPU_Computing_SDK/C/common/inc/
SDKLIB = /home/daroczyb/NVIDIA_GPU_Computing_SDK/C/lib/
SSEOPT= -msse3 -mfpmath=sse -funroll-loops
ARCH_GPU= sm_10
CFLAGS = $(OPT) $(SSEOPT)
#$(OPENCV_LINUX)

main: main.cpp OpenCL_env.o
	g++ $(CFLAGS) main.cpp OpenCL_env.o -L$(OPENCL) -I$(OPENCL_INC) -o SimKerCL
	
mac: main.cpp OpenCL_mac.o
	clang++ $(CFLAGS) $(OPENCL_MAC) main.cpp OpenCL_mac.o -o SimKerCL_mac

climage.o: image_core/climage.h image_core/climage.cpp
	g++ $(CFLAGS) `pkg-config --cflags opencv` image_core/climage.cpp `pkg-config --libs opencv` -c
#	g++ $(CFLAGS) -c image_core/climage.cpp 

OpenCL_env.o: OpenCL_env.cpp OpenCL_env.h
	g++ $(CFLAGS) -Wunused-result -I$(OPENCL_INC) -c OpenCL_env.cpp

OpenCL_mac.o: OpenCL_env.cpp OpenCL_env.h
	g++ $(OPT) $(OPENCL_MAC) -c OpenCL_env.cpp -o OpenCL_mac.o

clean:
	-rm -f *.o *.linkinfo *.so *.a proba
