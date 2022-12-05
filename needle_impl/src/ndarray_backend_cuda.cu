#include <cuda_runtime.h>
#include <cuComplex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);
typedef ssize_t ptrdiff_t;

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides




__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  ssize_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  if (gid < size) {
    const size_t shape_size = shape.size;
    ssize_t a_index = gid;

    ssize_t shape_factors[MAX_VEC_SIZE]; 
    shape_factors[shape_size - 1] = 1;
    for (size_t i = shape_size - 1; i != 0; --i) {
        shape_factors[i - 1] = shape_factors[i] * shape.data[i];
    }   

    ssize_t a_offset = offset;
    for (size_t i = 0; a_index != 0; ++i) {
      a_offset += strides.data[i] * (a_index / shape_factors[i]);
      a_index %= shape_factors[i];
    }   

    out[gid] = a[a_offset];
  }
  /// END YOUR SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                                   CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  if (gid < size) {
    const size_t shape_size = shape.size;
    size_t out_index = gid;

    size_t shape_factors[MAX_VEC_SIZE]; 
    shape_factors[shape_size - 1] = 1;
    for (size_t i = shape_size - 1; i != 0; --i) {
        shape_factors[i - 1] = shape_factors[i] * shape.data[i];
    }   

    size_t out_offset = offset;
    for (size_t i = 0; out_index != 0; ++i) {
      out_offset += strides.data[i] * (out_index / shape_factors[i]);
      out_index %= shape_factors[i];
    }

    out[out_offset] = a[gid];
  }
  /// END YOUR SOLUTION
}


void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                              VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}


__global__ void ScalarSetitemKernel(const scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                                    CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  if (gid < size) {
    const size_t shape_size = shape.size;
    size_t out_index = gid;

    size_t shape_factors[MAX_VEC_SIZE]; 
    shape_factors[shape_size - 1] = 1;
    for (size_t i = shape_size - 1; i != 0; --i) {
        shape_factors[i - 1] = shape_factors[i] * shape.data[i];
    }   

    size_t out_offset = offset;
    for (size_t i = 0; out_index != 0; ++i) {
      out_offset += strides.data[i] * (out_index / shape_factors[i]);
      out_index %= shape_factors[i];
    }

    out[out_offset] = val;
  }
  /// END YOUR SOLUTION
}


void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                               VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN YOUR SOLUTION
#define NEEDLE_EWISE_FN( X, A, B, O, I, Y )  \
 __global__ void Ewise##X##Kernel (const scalar_t* ( A ), const scalar_t* ( B ), scalar_t* ( O ), size_t size) { \
   size_t ( I ) = blockIdx.x * blockDim.x + threadIdx.x; \
   if (( I ) < size) { (( O )[( I )]) = ( Y ); } \
 } \
 void Ewise##X (const CudaArray& ( A ), const CudaArray& ( B ), CudaArray* ( O )) { \
   CudaDims dim = CudaOneDim(( O )->size); \
   Ewise##X##Kernel <<<dim.grid, dim.block>>>(( A ).ptr, ( B ).ptr, ( O )->ptr, ( O )->size); \
 }

#define NEEDLE_SCALAR_FN( X, A, B, O, I, Y )  \
 __global__ void Scalar##X##Kernel (const scalar_t* ( A ), scalar_t ( B ), scalar_t* ( O ), size_t size) { \
   size_t ( I ) = blockIdx.x * blockDim.x + threadIdx.x; \
   if (( I ) < size) { (( O )[( I )]) = ( Y ); } \
 } \
 void Scalar##X (const CudaArray& ( A ), scalar_t ( B ), CudaArray* ( O )) { \
   CudaDims dim = CudaOneDim(( O )->size); \
   Scalar##X##Kernel <<<dim.grid, dim.block>>>(( A ).ptr, ( B ), ( O )->ptr, ( O )->size); \
 }

#define NEEDLE_SINGLE_EWISE_FN( X, A, O, I, Y )  \
 __global__ void Ewise##X##Kernel (const scalar_t* ( A ), scalar_t* ( O ), size_t size) { \
   size_t ( I ) = blockIdx.x * blockDim.x + threadIdx.x; \
   if (( I ) < size) { (( O )[( I )]) = ( Y ); } \
 } \
 void Ewise##X (const CudaArray& ( A ), CudaArray* ( O )) { \
   CudaDims dim = CudaOneDim(( O )->size); \
   Ewise##X##Kernel <<<dim.grid, dim.block>>>(( A ).ptr, ( O )->ptr, ( O )->size); \
 }

NEEDLE_EWISE_FN(Mul, a, b, out, gid, a[gid] * b[gid]);
NEEDLE_SCALAR_FN(Mul, a, val, out, gid, a[gid] * val);
NEEDLE_EWISE_FN(Div, a, b, out, gid, a[gid] / b[gid]);
NEEDLE_SCALAR_FN(Div, a, val, out, gid, a[gid] / val);
NEEDLE_SCALAR_FN(Power, a, val, out, gid, pow(a[gid], val));
NEEDLE_EWISE_FN(Maximum, a, b, out, gid, (a[gid] > b[gid] ? a[gid] : b[gid]));
NEEDLE_SCALAR_FN(Maximum, a, val, out, gid, (a[gid] > val ? a[gid] : val));
NEEDLE_EWISE_FN(Eq, a, b, out, gid, a[gid] == b[gid]);
NEEDLE_SCALAR_FN(Eq, a, val, out, gid, a[gid] == val);
NEEDLE_EWISE_FN(Ge, a, b, out, gid, a[gid] >= b[gid]);
NEEDLE_SCALAR_FN(Ge, a, val, out, gid, a[gid] >= val);
NEEDLE_SINGLE_EWISE_FN(Log, a, out, gid, log(a[gid]));
NEEDLE_SINGLE_EWISE_FN(Exp, a, out, gid, exp(a[gid]));
NEEDLE_SINGLE_EWISE_FN(Tanh, a, out, gid, tanh(a[gid]));
/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out,
                             uint32_t M, uint32_t N, uint32_t P) {
  /**
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  // convert this to row and col
  // gid = row * P + col
  size_t j = gid % P;
  size_t i = gid / P;
  if (i < M && j < P) {
    // out[i, j] = sum_k A[i, k] * B[k, j]
    scalar_t elem = 0;
    for (size_t k = 0; k < N; ++k) {
      elem += a[i * N + k] * b[k * P + j]; // A[i, k] * B[k, j]
    }
    out[i * P + j] = elem; // out[i, j]
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(M * P);
  MatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END YOUR SOLUTION
}

__global__ void ForwardFourier1DKernel(const scalar_t* a_real, const scalar_t* a_imag, scalar_t* out_real, scalar_t* out_imag,
                                       uint32_t n) {
  /*
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  size_t j = gid % n;
  size_t k = gid / n;
  if (k < n && j < n) {
    cuDoubleComplex c;
    auto a_j = make_cuDoubleComplex(a[j], 0.0);
    double theta = -2.0 * j * k / n;
    sincospi(theta, &(c.y), &(c.x));
    c = cuCmul(c, a_j);
    printf("%lu %lu %f %f\n", j, k, c.x, c.y);
    __syncthreads();
    out_real[k] += __double2float_rn(c.x);
    out_imag[k] += __double2float_rn(c.y);
    __syncthreads();
  }
  */

  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < n) {
    cuDoubleComplex c;
    double real = 0.0;
    double imag = 0.0;

    for (size_t j = 0; j < n; ++j) {
      auto a_j = make_cuDoubleComplex(a_real[j], (a_imag != nullptr) ? a_imag[j] : 0.0);
      double theta = -2.0 * j * gid / n;
      sincospi(theta, &(c.y), &(c.x));
      c = cuCmul(c, a_j);
      real += c.x;
      imag += c.y;
    }

    out_real[gid] = __double2float_rn(real);
    out_imag[gid] = __double2float_rn(imag);
  }
}

void ForwardFourierReal1D(const CudaArray& a, CudaArray* out_real, CudaArray* out_imag, uint32_t n) {
  // CudaDims dim = CudaOneDim(n * n);
  CudaDims dim = CudaOneDim(n);
  ForwardFourier1DKernel<<<dim.grid, dim.block>>>(a.ptr, nullptr, out_real->ptr, out_imag->ptr, n);
}

void ForwardFourierComplex1D(const CudaArray& a_real, const CudaArray& a_imag, CudaArray* out_real, CudaArray* out_imag, uint32_t n) {
  // CudaDims dim = CudaOneDim(n * n);
  CudaDims dim = CudaOneDim(n);
  ForwardFourier1DKernel<<<dim.grid, dim.block>>>(a_real.ptr, a_imag.ptr, out_real->ptr, out_imag->ptr, n);
}

__global__ void BackwardFourier1DKernel(const scalar_t* a_real, const scalar_t* a_imag, scalar_t* out_real, scalar_t* out_imag,
                                        uint32_t n) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < n) {
    cuDoubleComplex c;
    double real = 0.0;
    double imag = 0.0;

    for (size_t j = 0; j < n; ++j) {
      auto a_j = make_cuDoubleComplex(a_real[j], (a_imag != nullptr) ? a_imag[j] : 0.0);
      double theta = 2.0 * j * gid / n;
      sincospi(theta, &(c.y), &(c.x));
      c = cuCmul(c, a_j);
      real += c.x;
      imag += c.y;
    }

    out_real[gid] = __double2float_rn(real);
    out_imag[gid] = __double2float_rn(imag);
  }
}

void BackwardFourierReal1D(const CudaArray& a, CudaArray* out_real, CudaArray* out_imag, uint32_t n) {
  CudaDims dim = CudaOneDim(n);
  BackwardFourier1DKernel<<<dim.grid, dim.block>>>(a.ptr, nullptr, out_real->ptr, out_imag->ptr, n);
}

void BackwardFourierComplex1D(const CudaArray& a_real, const CudaArray& a_imag, CudaArray* out_real, CudaArray* out_imag, uint32_t n) {
  CudaDims dim = CudaOneDim(n);
  BackwardFourier1DKernel<<<dim.grid, dim.block>>>(a_real.ptr, a_imag.ptr, out_real->ptr, out_imag->ptr, n);
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = a[gid * reduce_size];
    for (size_t i = gid * reduce_size + 1; i < (gid + 1) * reduce_size; ++i) {
      if (a[i] > out[gid]) {
        out[gid] = a[i];
      }
    }
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(a.size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END YOUR SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = a[gid * reduce_size];
    for (size_t i = gid * reduce_size + 1; i < (gid + 1) * reduce_size; ++i) {
      out[gid] += a[i];
    }
  }
}


void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(a.size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END YOUR SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("naive_forward_fourier_real_1d", ForwardFourierReal1D);
  m.def("naive_forward_fourier_complex_1d", ForwardFourierComplex1D);
  m.def("naive_backward_fourier_real_1d", BackwardFourierReal1D);
  m.def("naive_backward_fourier_complex_1d", BackwardFourierComplex1D);
  m.def("fast_forward_fourier_real_1d", ForwardFourierReal1D);  // naive FT is parallelized anyway
  m.def("fast_forward_fourier_complex_1d", ForwardFourierComplex1D);
  m.def("fast_backward_fourier_real_1d", BackwardFourierReal1D);
  m.def("fast_backward_fourier_complex_1d", BackwardFourierComplex1D);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
