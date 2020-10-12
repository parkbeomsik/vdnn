#include <stdio.h>
#include <string.h>

#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include <iostream>
#include <vector>

#include <cublas_v2.h>
// #include <cudnn.h>
#include "/home/beomsik/cuda/include/cudnn.h"
//#include <cnmem.h>
// #include "/home/mrhu/vdnn/cnmem/include/cnmem.h"
#include "/home/beomsik/vdnn/cnmem/include/cnmem.h"

#include <unistd.h>
#include <time.h>
#include <pthread.h>


#ifdef USE_CPP_11
#include <thread>
#endif


//-------------------------------
// profiling individual layers
//#define _PROFILE_INDIVIDUAL_LAYERS_ 1
#define _PROFILE_FWD_ 1 
#define _PROFILE_BWD_ 0
int _PROFILE_LAYER_ID_  = 999;

//#define _DEBUG_ 1
//#define _ANALYSIS_ 1

#define _REUSE_DISTANCE_ 1
#define _USE_CNMEM_MEM_USAGE_ 1

#define _COMPUTE_CHECKSUM_ 1

#ifdef _COMPUTE_CHECKSUM_
#include <zlib.h>
#else
#define _PROFILE_CONV_ALGO_ 1
#endif

#define _INCLUDE_FC_LAYERS_ 1

// PCI-E (gen2) --  8 GB/s (with 80% effective bw)
#define OFFLOAD_PREFETCH_BW ((float)1024*1024* 8*0.8)
// PCI-E (gen3) -- 16 GB/s (with 80% effective bw)
//#define OFFLOAD_PREFETCH_BW ((float)1024*1024*16*0.8)

#define ASSERT_EQ(A, B) {  \
  if((A)!=(B)) { printf("\n\n[CNMEM FAILED]\n"); this->printCnmemMemoryUsage(); assert(0); }        \
}

#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}


#define checkCUDNN(status) {                                           \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << status;                           \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCUBLAS(status) {                                          \
    std::stringstream _error;                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                              \
      _error << "CUBLAS failure: " << status;                           \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      assert(0);                                                        \
      FatalError(_error.str());                                        \
    }                                                                  \
}


// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{

  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    //assert(result == cudaSuccess);
  }

  return result;
}

// mrhu
// change both below
#define value_type float
#define DATA_PRECISION  CUDNN_DATA_FLOAT

/*
#define value_type short
#define DATA_PRECISION  CUDNN_DATA_HALF
*/

typedef enum
{
    VDNN_NONE     = 0,
    VDNN_ALL      = 1,
    VDNN_CONV     = 2,
    VDNN_NUM_TYPES    ,
} vdnn_t;

typedef enum
{
  VDNN_MEMORY_OPT_ALGO  = 0 ,
  VDNN_PERF_OPT_ALGO    = 1 ,
  VDNN_DYNAMIC_ALGO     = 2 ,
  VDNN_NUM_ALGOMODE         ,
} vdnnAlgoMode_t;
vdnnAlgoMode_t AlgoMode;

typedef enum
{
    CONV            = 0,
    RELU            = 1,
    POOL            = 2,
    TANH            = 3,
    SIGMOID         = 4,
    FC_CONV         = 5,
    CONCATENATE     = 6,  // GOOGLENET
    FC_GEMM         = 7,
    NUM_LAYER_TYPES = 8,
} Layer_t;

typedef enum
{
  ALEXNET           = 0,
  OVERFEAT_FAST     = 1,
  OVERFEAT_ACCURATE = 2,
  VGG_008           = 3,
  VGG_013           = 4,
  VGG_016           = 5,
  VGG_116           = 6,  // mrhu: you will be able to add as much cfg_X layers as possible until pinned memory allocation space overflows host memory
  VGG_216           = 7,
  VGG_316           = 8,
  VGG_416           = 9,
  VGG_516           = 10,
  GOOGLENET         = 11,
  FC_ONLY           = 12,
  MNIST		    = 13,
  NUM_CNN_TYPE      ,
} CNN_t;
CNN_t cnnType;


class Layer {
  Layer(
      Layer_t         _layerType,
      cudnnHandle_t*  _cudnnHandle,
      cudaStream_t*   _stream_compute,
      cudaStream_t*   _stream_memory,
      int _id,
      int _n, int _c, int _h, int _w, 
      int _pad_h, int _pad_w, int _stride_h, int _stride_w, 
      int _k, int _r, int _s);
  Layer(
      Layer_t         _layerType,
      cublasHandle_t* _cublasHandle,
      cudaStream_t*   _stream_compute,
      cudaStream_t*   _stream_memory,
      int _id,
      int _n, int _c, int _h, int _w, 
      int _pad_h, int _pad_w, int _stride_h, int _stride_w, 
      int _k, int _r, int _s);

  ~Layer();   
  
  // copy pointers 
  void  copySrcData(void* _srcData);
  void  copyDstData(void* _dstData);
  void  copyDiffData(void* _diffData);
  void  copyGradData(void* _gradData);

  // intialize algo to mem-opt _without_ profiling
  float findMemOptFwdAlgo();
  float findMemOptBwdFilterAlgo();
  float findMemOptBwdDataAlgo();

  // changing layer algo
  void  switchFwdAlgo(unsigned _algoIndex);
  void  switchBwdFilterAlgo(unsigned _algoIndex);
  void  switchBwdDataAlgo(unsigned _algoIndex);

  // profiling 
  float findBestFwdAlgo(vdnnAlgoMode_t _algoMode);
  float findBestBwdFilterAlgo(vdnnAlgoMode_t _algoMode);
  float findBestBwdDataAlgo(vdnnAlgoMode_t _algoMode);
  void  print();
  size_t  printGpuMemoryUsage();
  size_t  printCnmemMemoryUsage();

  // computation
  //void  cudaFreePrevSrcData(void* _srcDataFromPrevLayerToFree);
  void  cudaMallocCurrDstData();

  // CONV layer
  size_t  convolutionForward(bool* _offloaded, value_type** _offloadedSrcData_h);
  size_t  convolutionBackward(int _layer_id_to_prefetch, void* _prefetchedSrcData_h, void* _srcDataToPrefetch, unsigned long _prefetch_bytes, size_t* _memUsedFilter, size_t* _memUsedData);

  // FC_GEMM layer
  size_t  fullyConnectedForward(bool* _offloaded, value_type** _offloadedSrcData_h);
  size_t  fullyConnectedBackward(int _layer_id_to_prefetch, void* _prefetchedSrcData_h, void* _srcDataToPrefetch, unsigned long _prefetch_bytes, size_t* _memUsedFilter, size_t* _memUsedData);

  // RELU layer
  size_t  activationForward(bool* _offloaded, value_type** _offloadedSrcData_h);
  size_t  activationBackward(int _layer_id_to_prefetch, void* _prefetchedSrcData_h, void* _srcDataToPrefetch, unsigned long _prefetch_bytes);

  // POOL layer
  size_t  poolingForward(bool* _offloaded, value_type** _offloadedSrcData_h);
  size_t  poolingBackward(int _layer_id_to_prefetch, void* _prefetchedSrcData_h, void* _srcDataToPrefetch, unsigned long _prefetch_bytes);

  // CONCATENATE layer
  size_t  concatenateForward();
  size_t  concatenateBackward();

  // gets
  Layer_t       LayerType()   { assert(layerType<NUM_LAYER_TYPES); return layerType; }
  void*         SrcData()     { return srcData;                 }
  void*         DstData()     { return dstData;                 }
  void*         GradData()    { return gradData;                }
  int           SrcDataSize() { return n_in*c_in*h_in*w_in;     }
  int           DstDataSize() { return n_out*c_out*h_out*w_out; }
  int           FilterDataSize()  { return k*c_in*r*s;          }


  // mem usage
  size_t        memUsageSrcData();
  size_t        memUsageDstData();    
  size_t        memUsageFilterData();  
  size_t        memUsageDiffData(); 
  size_t        memUsageGradData();
  size_t        memUsageWorkSpaceFwd();
  size_t        memUsageWorkSpaceBwd();

  size_t        deriveMemUsageFwd();
  size_t        deriveMemUsageBwd();
  size_t        deriveMemUsageFwdLayerWise();
  size_t        deriveMemUsageBwdLayerWise();

 
  void          decrementRefCntFwd()  { assert(refCntFwd>0);  refCntFwd--;  }
  void          decrementRefCntBwd()  { assert(refCntBwd>0);  refCntBwd--;  }
private:
  Layer_t   layerType;
  int       id;

  //--------
  // input
  //--------
  // input NCHW
  int       n_in;
  int       c_in;
  int       h_in;
  int       w_in;
  // padding, stride
  int       pad_h;
  int       pad_w;
  int       stride_h;
  int       stride_w;
  int       k;
  int       r;
  int       s;
  //--------
  // output
  //--------
  // output NCHW
  int       n_out;
  int       c_out;
  int       h_out;
  int       w_out;

  // streams
  cudaStream_t* 	        stream_compute;
	cudaStream_t*	          stream_memory;

  //------------
  // ptrs to memory
  //------------
  void*     srcData;
  void*     filterData;
  void*     biasData;
  void*     dstData;
  void*     diffData;
  void*     gradData;
  
  // for backprop
  //value_type*   diffData; // backprop input to layer
  //value_type*   gradData; // backprop output of layer for 1) backpropData (when forwarding to previous layer) or 2) backpropFilter (to update wGrad within this layer)

  //--------
  // cuDNN 
  //--------
  cudnnHandle_t*                cudnnHandle;  // shared across all layers
  cudnnDataType_t               dataType;
  cudnnConvolutionMode_t        modeConv;
  cudnnTensorFormat_t           tensorFormat;
  cudnnTensorDescriptor_t       srcTensorDesc, dstTensorDesc, biasTensorDesc;
  cudnnTensorDescriptor_t       srcDiffTensorDesc, dstDiffTensorDesc;   // backprop
  cudnnFilterDescriptor_t       filterDesc;
  cudnnConvolutionDescriptor_t  convDesc;
  cudnnPoolingDescriptor_t      poolingDesc;
  bool                          inPlaceOp;

  // profile results (chosen)
  ProfiledFwdAlgoPerf_t         fwdAlgoPerf;
  ProfiledBwdDataAlgoPerf_t     bwdDataAlgoPerf;
  ProfiledBwdFilterAlgoPerf_t   bwdFilterAlgoPerf;

  // profile results (all)
  cudnnConvolutionFwdAlgoPerf_t*        fwdProfileResults;
  cudnnConvolutionBwdFilterAlgoPerf_t*  bwdFilterProfileResults;
  cudnnConvolutionBwdDataAlgoPerf_t*    bwdDataProfileResults;

  int                                   profiledFwdAlgoCount;
  int                                   profiledBwdFilterAlgoCount;
  int                                   profiledBwdDataAlgoCount;

  //--------
  // cuBLAS
  //--------
  cublasHandle_t*               cublasHandle;
  //cublasOperation_t             cublasOp;

  // GOOGLENET
  bool  insideInceptionModule;
  bool  headOfFork;
  bool  tailOfFork;
  int   forkId;
  int   concatenateChannelOffset;
  std::vector<int>  producerLayerId;
  std::vector<int>  consumerLayerId;

  int   refCntFwd;
  int   refCntBwd;

friend struct Network;
};


Layer::Layer(
      Layer_t         _layerType,
      cudnnHandle_t*  _cudnnHandle,
      cudaStream_t*   _stream_compute,
      cudaStream_t*   _stream_memory,
      int _id,
      int _n, int _c, int _h, int _w, 
      int _pad_h, int _pad_w, int _stride_h, int _stride_w, 
      int _k, int _r, int _s):  layerType(_layerType), 
                                cudnnHandle(_cudnnHandle), stream_compute(_stream_compute), stream_memory(_stream_memory), 
                                id(_id),
                                n_in(_n), c_in(_c), h_in(_h), w_in(_w),
                                pad_h(_pad_h), pad_w(_pad_w), stride_h(_stride_h), stride_w(_stride_w),
                                k(_k), r(_r), s(_s)
{
  // GOOGLENET
  insideInceptionModule = false;
  headOfFork            = false;
  tailOfFork            = false;
  forkId                = -1;
  concatenateChannelOffset = -1;
  producerLayerId.clear();
  consumerLayerId.clear();
  refCntFwd             = 0;
  refCntBwd             = 0;


  // default    
  dataType      = DATA_PRECISION;
//  modeConv      = CUDNN_CONVOLUTION;
  modeConv      = CUDNN_CROSS_CORRELATION;
  tensorFormat  = CUDNN_TENSOR_NCHW;
  inPlaceOp     = false; 
  // No srcData cudaMalloc() when initializing Layer() 
  srcData       = NULL;
  filterData    = NULL;
  biasData      = NULL;
  dstData       = NULL;
  diffData      = NULL; // input to layer when backprop
  gradData      = NULL; // output of layer when backprop

  switch(_layerType) {
    case FC_GEMM:
      { 
        printf("Shouldn't be here ...\n");
        assert(0);
      }
      break;
    case CONV: 
    case FC_CONV:
      {
        // create cudnn descriptors
        checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    		value_type *filterData_h	= (value_type*)malloc(k*c_in*r*s*sizeof(value_type));
		value_type *biasData_h		= (value_type*)malloc(k*sizeof(value_type));
 
      	for(unsigned i=0; i<k*c_in*r*s; i++)	filterData_h[i]	= (value_type)i/(k*c_in*r*s);
	for(unsigned i=0; i<k; i++)		biasData_h[i]	= (value_type)i/(k);
        ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&filterData,  k*c_in*r*s*sizeof(value_type), NULL));
        ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&biasData, k*sizeof(value_type), NULL)); 

    		checkCuda(cudaMemcpy(filterData, filterData_h, k*c_in*r*s*sizeof(value_type), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(biasData, biasData_h, k*sizeof(value_type), cudaMemcpyHostToDevice));
        // free memory
        free(filterData_h);

        // set input descriptors
        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, tensorFormat, dataType, n_in, c_in, h_in, w_in));
      	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, dataType, k, c_in, r, s));
        checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc, tensorFormat, dataType, 1, k, 1, 1));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, stride_w, stride_h, 1, 1, modeConv));

        // find dimension of convolution output
        checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, srcTensorDesc, filterDesc, &n_out, &c_out, &h_out, &w_out));

        // set output descriptor based on above
        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, tensorFormat, dataType, n_out, c_out, h_out, w_out));


#ifdef _MRHU_
          printf("\n[Layer-%2d]\n", id);
          printf("R=%d S=%d C_in=%d C_out=%d K=%d\n", r, s, c_in, c_out, k);
          printf("FilterSize=%ld\n", this->memUsageFilterData());
          printf("n_in=%d c_in=%d h_in=%d w_in=%d\n", n_in, c_in, h_in, w_in);
          printf("SrcData = %ld\n", this->memUsageSrcData());
          printf("n_out=%d c_out=%d h_out=%d w_out=%d\n", n_out, c_out, h_out, w_out);
          printf("DstData = %ld\n", this->memUsageDstData());


          assert( (r*s*c_in*c_out)*sizeof(dataType)==this->memUsageFilterData() );

          assert( (h_in*w_in*c_in*n_in)*sizeof(dataType)==this->memUsageSrcData() );

          assert( (h_out*w_out*c_out*n_in)*sizeof(dataType)==this->memUsageDstData() );

          assert(c_out==k);
          assert(n_in==n_out);
#endif
      }
      break;
    case RELU:
        // create cudnn descriptors
        checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));

        // initialize n_/c_/h_/w_out
        n_out = n_in;
        c_out = c_in;
        h_out = h_in;
        w_out = w_in;

        // set input descriptors
        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, tensorFormat, dataType, n_in, c_in, h_in, w_in));

        // set output descriptor based on above
        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, tensorFormat, dataType, n_out, c_out, h_out, w_out));
        // default behavior of RELU is to do in-place op
        this->inPlaceOp = true;
      break;
    case CONCATENATE:
        // create cudnn descriptors
        checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));

        // initialize n_/c_/h_/w_out
        n_out = n_in;
        c_out = c_in;
        h_out = h_in;
        w_out = w_in;

        // set input descriptors
        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, tensorFormat, dataType, n_in, c_in, h_in, w_in));

        // set output descriptor based on above
        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, tensorFormat, dataType, n_out, c_out, h_out, w_out));
        // default behavior of CONCATENATE is to do in-place op
        this->inPlaceOp = true;

      break;
    case POOL:
        // create cudnn descriptors
        checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
        checkCUDNN(cudnnCreatePoolingDescriptor(&poolingDesc));

        // set input descriptors
        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, tensorFormat, dataType, n_in, c_in, h_in, w_in));
        checkCUDNN(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, r, s, pad_h, pad_w, stride_h, stride_w));
        // find dimension of pooling output
        checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolingDesc, srcTensorDesc, &n_out, &c_out, &h_out, &w_out));

        // set output descriptor based on above
        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, tensorFormat, dataType, n_out, c_out, h_out, w_out));
      break;

    default:
      assert(0);
      break;
  }

#ifdef _DEBUG_
  printf("Layer-%2d definition (Type=%d)\n", id, layerType);
  printf("Input:  %4d %4d %4d %4d\n", n_in, c_in, h_in, w_in);
  printf("Output: %4d %4d %4d %4d\n\n", n_out, c_out, h_out, w_out);
#endif
  // create descriptor for backprop
  checkCUDNN(cudnnCreateTensorDescriptor(&srcDiffTensorDesc));
  checkCUDNN(cudnnCreateTensorDescriptor(&dstDiffTensorDesc));
}

Layer::Layer(
      Layer_t         _layerType,
      cublasHandle_t* _cublasHandle,
      cudaStream_t*   _stream_compute,
      cudaStream_t*   _stream_memory,
      int _id,
      int _n, int _c, int _h, int _w, 
      int _pad_h, int _pad_w, int _stride_h, int _stride_w, 
      int _k, int _r, int _s):  layerType(_layerType), 
                                cublasHandle(_cublasHandle), stream_compute(_stream_compute), stream_memory(_stream_memory), 
                                id(_id),
                                n_in(_n), c_in(_c), h_in(_h), w_in(_w),
                                pad_h(_pad_h), pad_w(_pad_w), stride_h(_stride_h), stride_w(_stride_w),
                                k(_k), r(_r), s(_s)
{
  assert(_layerType==FC_GEMM);
  // GOOGLENET
  insideInceptionModule = false;
  headOfFork            = false;
  tailOfFork            = false;
  forkId                = -1;
  producerLayerId.clear();
  consumerLayerId.clear();
  refCntFwd             = 0;
  refCntBwd             = 0;


  // default    
  dataType      = DATA_PRECISION;
//  modeConv      = CUDNN_CONVOLUTION;
  modeConv      = CUDNN_CROSS_CORRELATION;
  tensorFormat  = CUDNN_TENSOR_NCHW;
  inPlaceOp     = false; 
  // No srcData cudaMalloc() when initializing Layer() 
  srcData       = NULL;
  filterData    = NULL;
  biasData      = NULL;
  dstData       = NULL;
  diffData      = NULL; // input to layer when backprop
  gradData      = NULL; // output of layer when backprop


  // malloc for layer weights + biases
  value_type *filterData_h	= (value_type*)malloc(k*c_in*r*s*sizeof(value_type));
  value_type *biasData_h	= (value_type*)malloc(k*sizeof(value_type));
 
  for(unsigned i=0; i<k*c_in*r*s; i++)	filterData_h[i]	= (value_type)i/(k*c_in*r*s);
  for(unsigned i=0; i<k; i++)		biasData_h[i]	= (value_type)i/(k);
  ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&filterData,  k*c_in*r*s*sizeof(value_type), NULL));
  ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&biasData, k*sizeof(value_type), NULL)); 

  checkCuda(cudaMemcpy(filterData, filterData_h, k*c_in*r*s*sizeof(value_type), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(biasData, biasData_h, k*sizeof(value_type), cudaMemcpyHostToDevice));
  // free memory
  free(filterData_h);

  // dimension of FC layer output
  // checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, srcTensorDesc, filterDesc, &n_out, &c_out, &h_out, &w_out));
  n_out = _n;
  c_out = _k;
  h_out = 1;
  w_out = 1;

#ifdef _DEBUG_
  printf("\n[Layer-%2d]\n", id);
  printf("R=%d S=%d C_in=%d C_out=%d K=%d\n", r, s, c_in, c_out, k);
  printf("FilterSize=%ld\n", this->memUsageFilterData());
  printf("n_in=%d c_in=%d h_in=%d w_in=%d\n", n_in, c_in, h_in, w_in);
  printf("SrcData = %ld\n", this->memUsageSrcData());
  printf("n_out=%d c_out=%d h_out=%d w_out=%d\n", n_out, c_out, h_out, w_out);
  printf("DstData = %ld\n", this->memUsageDstData());


  assert( (r*s*c_in*c_out)*sizeof(dataType)==this->memUsageFilterData() );

  assert( (h_in*w_in*c_in*n_in)*sizeof(dataType)==this->memUsageSrcData() );

  assert( (h_out*w_out*c_out*n_in)*sizeof(dataType)==this->memUsageDstData() );

  assert(c_out==k);
  assert(n_in==n_out);
#endif

#ifdef _DEBUG_
  printf("Layer-%2d definition (Type=%d)\n", id, layerType);
  printf("Input:  %4d %4d %4d %4d\n", n_in, c_in, h_in, w_in);
  printf("Output: %4d %4d %4d %4d\n\n", n_out, c_out, h_out, w_out);
#endif
}


size_t Layer::fullyConnectedForward(bool* _offloaded, value_type** _offloadedSrcData_h) {
  // sanity check: make sure all input/filter/output memory is allocated
  assert(srcData!=NULL);
  assert(filterData!=NULL);
  assert(dstData!=NULL);

  /*
    m: number of rows of matrix op(A) and C.
    n: number of columns of matrix op(B) and C.
    k: number of columns of op(A) and rows of op(B).

    lda: leading dimension of two-dimensional array used to store the matrix A.
    ldb: leading dimension of two-dimensional array used to store matrix B.
    ldc: leading dimension of a two-dimensional array used to store the matrix C.

    [NN]
                k                     n                       n
                _____                 ____                   _____
    m (=lda)  |       | x  k (=ldb) |      |  ==  m (=ldc)  |     |
              |       |             |      |                |     |
                -----                ------                  -----
  */

  /*
    M: batch
    N: output C
    K: input C
  */
  assert(n_in==n_out);
  assert(h_out==1);
  assert(w_out==1);
  assert((c_out*h_out*w_out)==c_out);

  int m   = n_in;
  int k   = c_in*h_in*w_in;
  int n   = c_out*h_out*w_out;
  int lda = m;
  int ldb = k;
  int ldc = m;

#ifdef _DEBUG_
  printf("[FWDPROP]");
#endif
  size_t  memUsed = printGpuMemoryUsage();

	// Do forward propagation
  value_type alpha = value_type(1);
  value_type beta  = value_type(0);

#ifdef _PROFILE_INDIVIDUAL_LAYERS_
  if( (_PROFILE_FWD_==1)&&(this->id==_PROFILE_LAYER_ID_) )
#endif
  checkCUBLAS(cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, (const float*)srcData/*A(m:k)*/, lda, (const float*)filterData/*B(k:n)*/, ldb, &beta, (float*)dstData, ldc));  

  // kick off async memcpy(d->h) of the input activations
  assert(_offloaded[id]==false);  // FC_GEMM never offloads
  if(_offloaded[id]==true) { 
    unsigned long  	offload_bytes = this->SrcDataSize()*sizeof(value_type);

    // launch async offloading to sysmem (pinned memory)
    checkCuda(cudaMemcpyAsync(&_offloadedSrcData_h[id][0], &(((value_type*)srcData)[0]), offload_bytes, cudaMemcpyDeviceToHost, *stream_memory));
  }

  // wait for compute to complete
  checkCuda(cudaStreamSynchronize(*stream_compute));

  return memUsed;


}

size_t Layer::convolutionForward(bool* _offloaded, value_type** _offloadedSrcData_h) {
  // sanity check: make sure all input/filter/output memory is allocated
  assert(srcData!=NULL);
  assert(filterData!=NULL);
  assert(dstData!=NULL);

  // check if workspace is needed for fwdprop
  void* workSpace=NULL;
  size_t sizeInBytes=0;

  // try to find the algo that fits within current mem budget
  unsigned algoIndex = 0;
  cnmemStatus_t cnmemMallocResult = CNMEM_STATUS_OUT_OF_MEMORY;
  do {
    // derive workspace size
	  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(*cudnnHandle,
                                            srcTensorDesc,
                                            filterDesc,
                                            convDesc,
                                            dstTensorDesc,
                                            fwdAlgoPerf.algo,
                                            &sizeInBytes));
  
    // if additional workspace is needed, cudaMalloc
    if (sizeInBytes!=0) {
      cnmemMallocResult = cnmemMalloc(&workSpace, sizeInBytes, NULL);
    }
    // no workspace required for this algo, so break out of the loop
    else {
      cnmemMallocResult = CNMEM_STATUS_SUCCESS;
      break;
    }

    // if malloc failed, switch to a less memory-hungry algo ..
    if(cnmemMallocResult!=CNMEM_STATUS_SUCCESS) {
      // if not dynamic algo mode, you only get one chance ...
      if(AlgoMode!=VDNN_DYNAMIC_ALGO) {
        printf("[ERROR] Failed allocating workspace ...\n");
        assert(0);
      }

      // already reached maximum algo available without being able to find one works ... (debug!)
      if(algoIndex==(this->profiledFwdAlgoCount-1)) {
        printf("Couldn't find algo ...\n");
        assert(0);
      }
#ifdef _DEBUG_
      printf("\n--------\nFailed algo\n");
      printf("Failed algo = %d (MemReq=%ld, Time=%lf)\n\n", fwdAlgoPerf.algo, fwdAlgoPerf.memory, fwdAlgoPerf.time);
#endif  
      for(unsigned currIdx=(algoIndex+1); currIdx<this->profiledFwdAlgoCount; currIdx++) {
#ifdef _DEBUG_
        printf("Next idx = %d\n", currIdx);
        printf("Next algo = %d (MemReq=%ld, Time=%lf)\n", fwdProfileResults[currIdx].algo, fwdProfileResults[currIdx].memory, fwdProfileResults[currIdx].time);
#endif
        // found next algo that 'CUDNN_STATUS_SUCCESS'
        if(fwdProfileResults[currIdx].status==CUDNN_STATUS_SUCCESS) {
          // update algoIndex
          algoIndex = currIdx;
          // switch algo
          this->switchFwdAlgo(algoIndex);
  
          // sanity
          assert(fwdAlgoPerf.time!=(float)-1);
          assert(fwdAlgoPerf.status==CUDNN_STATUS_SUCCESS);
          break;
        }
      }
#ifdef _DEBUG_
      printf("\n--------\nNext algo\n");
      printf("Next algo = %d (MemReq=%ld, Time=%lf)\n\n", fwdAlgoPerf.algo, fwdAlgoPerf.memory, fwdAlgoPerf.time);
#endif
    }
  } while(cnmemMallocResult!=CNMEM_STATUS_SUCCESS);
  // sanity 
  assert(cnmemMallocResult==CNMEM_STATUS_SUCCESS);

#ifdef _CSV_
  assert(fwdProfileResults[0].time<=this->fwdAlgoPerf.time);  // fastest algo should always be gt/eq to currently selected algo
  assert(fwdProfileResults[0].time!=-1);
  assert(this->fwdAlgoPerf.time!=-1);
  // calculate the latency overhead of choosing a less-performant algorithm (if we chose to do so)
  if(this->fwdAlgoPerf.time > fwdProfileResults[0].time) {
    statLatencyOverheadOfConvAlgoFwd += (this->fwdAlgoPerf.time - fwdProfileResults[0].time);
  }
#endif


#ifdef _DEBUG_
  printf("[FWDPROP]");
#endif
  size_t  memUsed = printGpuMemoryUsage();

	// Do forward propagation
  value_type alpha = value_type(1);
  value_type beta  = value_type(0);

#ifdef _PROFILE_INDIVIDUAL_LAYERS_
  if( (_PROFILE_FWD_==1)&&(this->id==_PROFILE_LAYER_ID_) )
#endif
	checkCUDNN(cudnnConvolutionForward(  *cudnnHandle,
                            &alpha,
                            srcTensorDesc,
                            srcData,
                            filterDesc,
                            filterData,
                            convDesc,
                            fwdAlgoPerf.algo,
                            workSpace,
                            sizeInBytes,
                            &beta,
                            dstTensorDesc,
                            dstData
                          ));

  // kick off async memcpy(d->h) of the input activations
  if(_offloaded[id]==true) { 
    unsigned long  	offload_bytes = this->SrcDataSize()*sizeof(value_type);

    // launch async offloading to sysmem (pinned memory)
    checkCuda(cudaMemcpyAsync(&_offloadedSrcData_h[id][0], &(((value_type*)srcData)[0]), offload_bytes, cudaMemcpyDeviceToHost, *stream_memory));
  }

  
  // add bias
#ifdef _PROFILE_INDIVIDUAL_LAYERS_
  if( (_PROFILE_FWD_==1)&&(this->id==_PROFILE_LAYER_ID_) )
#endif
  checkCUDNN(cudnnAddTensor(  *cudnnHandle, 
                              &alpha, 
                              biasTensorDesc,
                              biasData,
                              &alpha,
                              dstTensorDesc,
                              dstData
                            ));
  
  // free device memory allocated for workspace
  if (sizeInBytes!=0) {
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(workSpace, NULL)); 
  }

  // wait for compute to complete
  checkCuda(cudaStreamSynchronize(*stream_compute));

  return memUsed;
}

size_t Layer::convolutionBackward(int _layer_id_to_prefetch, void* _prefetchedSrcData_h, void* _srcDataToPrefetch, unsigned long _prefetch_bytes, size_t* _memUsedFilter, size_t* _memUsedData) {
  // sanity
  assert(srcData!=NULL);
  assert(dstData!=NULL);    // need to release this
  assert(gradData==NULL);   // RELU is an in-place algorithm, so no need to allocate diffOutput
  assert(diffData!=NULL);
  assert( (this->LayerType()==CONV)||(this->LayerType()==FC_CONV) );

  // Release memory for dstData -- we won't need this at conv layer (pointer clean-up will be done after this method is done)
  ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(this->dstData, NULL));

  // check if workspace is needed for bwdprop
  // NOTE) we will calculate the workspace required for both filter/data and allocate based on the larger of these two (to obviate the need for two malloc/frees)
  void* workSpace=NULL;
  size_t sizeInBytesFilter  = 0;
  //-------------------------------
  // 1) wgrad derivation
  //-------------------------------
  // try to find the algo that fits within current mem budget
  unsigned algoIndex = 0;
  cnmemStatus_t cnmemMallocResult = CNMEM_STATUS_OUT_OF_MEMORY;
  do {
    // derive workspace size (for wgrad)
    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(  *cudnnHandle,
                                                                srcTensorDesc,
                                                                dstTensorDesc,  // diffDesc
                                                                convDesc,
                                                                filterDesc,     // gradDesc
                                                                bwdFilterAlgoPerf.algo,  
                                                                &sizeInBytesFilter
                                                              ));


    // if additional workspace is needed, cudaMalloc
    if (sizeInBytesFilter!=0) {
      cnmemMallocResult = cnmemMalloc(&workSpace, sizeInBytesFilter, NULL);
    }
    // no workspace required for this algo, so break out of the loop
    else {
      cnmemMallocResult = CNMEM_STATUS_SUCCESS;
      break;
    }

    // if malloc failed, switch to a less memory-hungry algo ..
    if(cnmemMallocResult!=CNMEM_STATUS_SUCCESS) {
      // if not dynamic algo mode, you only get one chance ...
      if(AlgoMode!=VDNN_DYNAMIC_ALGO) {
#ifdef _DEBUG_
      printf("WD MALLOC FAIL with alloc size = %ld\n", sizeInBytesFilter);
#endif
        printf("[ERROR] Failed allocating workspace ...\n");
        assert(0);
      }

      // already reached maximum algo available without being able to find one works ... (debug!)
      if(algoIndex==(this->profiledBwdFilterAlgoCount-1)) {
        printf("Couldn't find algo ...\n");
        assert(0);
      }

      for(unsigned currIdx=(algoIndex+1); currIdx<this->profiledBwdFilterAlgoCount; currIdx++) {
        // found next algo that 'CUDNN_STATUS_SUCCESS'
        if(bwdFilterProfileResults[currIdx].status==CUDNN_STATUS_SUCCESS) {
          // update algoIndex
          algoIndex = currIdx;
          // switch algo
          this->switchBwdFilterAlgo(algoIndex);
  
          // sanity
          assert(bwdFilterAlgoPerf.time!=(float)-1);
          assert(bwdFilterAlgoPerf.status==CUDNN_STATUS_SUCCESS);
          break;
        }
      }
    }
  } while(cnmemMallocResult!=CNMEM_STATUS_SUCCESS);
  // sanity 
  assert(cnmemMallocResult==CNMEM_STATUS_SUCCESS);

#ifdef _CSV_
  assert(bwdFilterProfileResults[0].time<=this->bwdFilterAlgoPerf.time);  // fastest algo should always be gt/eq to currently selected algo
  assert(bwdFilterProfileResults[0].time!=-1);
  assert(this->bwdFilterAlgoPerf.time!=-1);
  // calculate the latency overhead of choosing a less-performant algorithm (if we chose to do so)
  if(this->bwdFilterAlgoPerf.time > bwdFilterProfileResults[0].time) {
    statLatencyOverheadOfConvAlgoBwdFilter += (this->bwdFilterAlgoPerf.time - bwdFilterProfileResults[0].time);
  }
#endif


  // find gradData size (for wgrad, dgrad) -- allocate memory based on max of two
  assert(this->FilterDataSize()>0);
  size_t  sizeFilterData  = this->FilterDataSize()*sizeof(value_type);
  ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&this->gradData, sizeFilterData, NULL));

  // gradient derivation 
  value_type alpha = value_type(1);
  value_type beta  = value_type(0);
  assert(this->gradData!=NULL);  

  //-------------------------------
  // 1) bgrad derivation
  //-------------------------------
#ifdef _PROFILE_INDIVIDUAL_LAYERS_
  if( (_PROFILE_BWD_==1)&&(this->id==_PROFILE_LAYER_ID_) )
#endif
  checkCUDNN(cudnnConvolutionBackwardBias(  *cudnnHandle,
                                            &alpha,
                                            dstTensorDesc,
                                            diffData,
                                            &beta,
                                            biasTensorDesc,
                                            gradData
                                            ));

#ifdef _DEBUG_
    printf("[BWDPROP][FILTER]");
#endif
    size_t  memUsedFilter = printGpuMemoryUsage();
    *_memUsedFilter = memUsedFilter;

  //-------------------------------
  // 2) wgrad derivation
  //-------------------------------
#ifdef _PROFILE_INDIVIDUAL_LAYERS_
  if( (_PROFILE_BWD_==1)&&(this->id==_PROFILE_LAYER_ID_) )
#endif
  checkCUDNN(cudnnConvolutionBackwardFilter_v3( *cudnnHandle,
                                                &alpha,
                                                srcTensorDesc,
                                                srcData,
                                                dstTensorDesc,
                                                diffData,
                                                convDesc,
                                                bwdFilterAlgoPerf.algo,
                                                workSpace,
                                                sizeInBytesFilter,
                                                &beta,
                                                filterDesc,
                                                gradData
                                              ));

  // free device memory allocated for workspace
  if (sizeInBytesFilter!=0) {
    assert(workSpace!=NULL);
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(workSpace, NULL));
  }

  // free device memory allocated for gradData
  ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(this->gradData, NULL));



  //-------------------------------
  // 3) dgrad derivation
  //-------------------------------
  workSpace       = NULL;
  this->gradData  = NULL;
  size_t sizeInBytesData    = 0;
  // try to find the algo that fits within current mem budget
  algoIndex = 0;
  cnmemMallocResult = CNMEM_STATUS_OUT_OF_MEMORY;
  do {
    // derive workspace size (for dgrad)
    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(  *cudnnHandle,
                                                              filterDesc,
                                                              dstTensorDesc,
                                                              convDesc,
                                                              srcTensorDesc,  // gradDesc
                                                              bwdDataAlgoPerf.algo,
                                                              &sizeInBytesData
                                                              ));

    // if additional workspace is needed, cudaMalloc
    if (sizeInBytesData!=0) {
      cnmemMallocResult = cnmemMalloc(&workSpace, sizeInBytesData, NULL);
    }
    // no workspace required for this algo, so break out of the loop
    else {
      cnmemMallocResult = CNMEM_STATUS_SUCCESS;
      break;
    }

    // if malloc failed, switch to a less memory-hungry algo ..
    if(cnmemMallocResult!=CNMEM_STATUS_SUCCESS) {
      // if not dynamic algo mode, you only get one chance ...
      if(AlgoMode!=VDNN_DYNAMIC_ALGO) {
        printf("[ERROR] Failed allocating workspace ...\n");
        assert(0);
      }

      // already reached maximum algo available without being able to find one works ... (debug!)
      if(algoIndex==(this->profiledBwdDataAlgoCount-1)) {
        printf("Couldn't find algo ...\n");
        assert(0);
      }

      for(unsigned currIdx=(algoIndex+1); currIdx<this->profiledBwdDataAlgoCount; currIdx++) {
        // found next algo that 'CUDNN_STATUS_SUCCESS'
        if(bwdDataProfileResults[currIdx].status==CUDNN_STATUS_SUCCESS) {
          // update algoIndex
          algoIndex = currIdx;
          // switch algo
          this->switchBwdDataAlgo(algoIndex);
  
          // sanity
          assert(bwdDataAlgoPerf.time!=(float)-1);
          assert(bwdDataAlgoPerf.status==CUDNN_STATUS_SUCCESS);
          break;
        }
      }
    }
  } while(cnmemMallocResult!=CNMEM_STATUS_SUCCESS);
  // sanity 
  assert(cnmemMallocResult==CNMEM_STATUS_SUCCESS);

#ifdef _CSV_
  assert(bwdDataProfileResults[0].time<=this->bwdDataAlgoPerf.time);  // fastest algo should always be gt/eq to currently selected algo
  assert(bwdDataProfileResults[0].time!=-1);
  assert(this->bwdDataAlgoPerf.time!=-1);
  // calculate the latency overhead of choosing a less-performant algorithm (if we chose to do so)
  if(this->bwdDataAlgoPerf.time > bwdDataProfileResults[0].time) {
    statLatencyOverheadOfConvAlgoBwdData += (this->bwdDataAlgoPerf.time - bwdDataProfileResults[0].time);
  }
#endif


  // find gradData size
  assert(this->SrcDataSize()>0);
  size_t  sizeSrcData     = this->SrcDataSize()*sizeof(value_type);
  ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&this->gradData, sizeSrcData, NULL));

#ifdef _DEBUG_
    printf("[BWDPROP][DATA]");
#endif
    size_t  memUsedData = printGpuMemoryUsage();
    *_memUsedData = memUsedData;

  // gradient derivation 
  assert(this->gradData!=NULL);  
#ifdef _PROFILE_INDIVIDUAL_LAYERS_
  if( (_PROFILE_BWD_==1)&&(this->id==_PROFILE_LAYER_ID_) )
#endif
  checkCUDNN(cudnnConvolutionBackwardData_v3( *cudnnHandle,
                                                &alpha,
                                                filterDesc,
                                                filterData,
                                                dstTensorDesc,
                                                diffData,
                                                convDesc,
                                                bwdDataAlgoPerf.algo,
                                                workSpace,
                                                sizeInBytesData,
                                                &beta,
                                                srcTensorDesc,  
                                                gradData
                                              ));

  // free device memory allocated for workspace
  if (sizeInBytesData!=0) {
    assert(workSpace!=NULL);
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(workSpace, NULL));
  }

  //-----------------------
  // prefetching
  //-----------------------
  // kick off async memcpy(h->d) to prefetch srcData
  if(_layer_id_to_prefetch!=-1) { 
    // sanity
    assert(_prefetchedSrcData_h!=NULL);
    assert(_srcDataToPrefetch!=NULL);
    assert(_prefetch_bytes>0);

    checkCuda(cudaMemcpyAsync(&(((value_type*)_srcDataToPrefetch)[0]), &(((value_type*)_prefetchedSrcData_h)[0]), _prefetch_bytes, cudaMemcpyHostToDevice, *stream_memory));
  }
  else {
    // sanity
    assert(_prefetchedSrcData_h==NULL);
    assert(_srcDataToPrefetch==NULL);
    assert(_prefetch_bytes==0);
  }

  // free diffData
  assert(this->diffData!=NULL);
  ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(this->diffData, NULL));


  // wait for compute to complete
  checkCuda(cudaStreamSynchronize(*stream_compute));

  size_t  memUsed = (memUsedFilter > memUsedData) ? memUsedFilter : memUsedData;
  return memUsed;
}

