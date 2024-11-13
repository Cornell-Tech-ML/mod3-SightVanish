# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py



## Diagnostics Output from parallel_check.py

>MAP
>
>================================================================================
> Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
>/Users/liwuchen/Library/Mobile 
>Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine 
>Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (163)  
>================================================================================
>
>
>Parallel loop listing for  Function tensor_map.<locals>._map, /Users/liwuchen/Library/Mobile Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (163) 
>----------------------------------------------------------------------------------------|loop #ID
>    def _map(                                                                           | 
>        out: Storage,                                                                   | 
>        out_shape: Shape,                                                               | 
>        out_strides: Strides,                                                           | 
>        in_storage: Storage,                                                            | 
>        in_shape: Shape,                                                                | 
>        in_strides: Strides,                                                            | 
>    ) -> None:                                                                          | 
>        stride_aligned = np.array_equal(out_strides, in_strides) and np.array_equal(    | 
>            out_shape, in_shape                                                         | 
>        )                                                                               | 
>        if stride_aligned:                                                              | 
>            for ordinal in prange(len(out)):--------------------------------------------| #2
>                # if strides are aligned, we can avoid indexing                         | 
>                out[ordinal] = fn(in_storage[ordinal])                                  | 
>        else:                                                                           | 
>            for ordinal in prange(len(out)):--------------------------------------------| #3
>                out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)-------------------| #0
>                in_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)--------------------| #1
>                # ordinal -> index                                                      | 
>                to_index(ordinal, out_shape, out_index)                                 | 
>                broadcast_index(out_index, out_shape, in_shape, in_index)               | 
>                # index -> real ordinal in memory                                       | 
>                o = index_to_position(out_index, out_strides)                           | 
>                j = index_to_position(in_index, in_strides)                             | 
>                out[o] = fn(in_storage[j])                                              | 
>--------------------------------- Fusing loops ---------------------------------
>Attempting fusion of parallel loops (combines loops with similar properties)...
>
>Fused loop summary:
>+--0 has the following loops fused into it:
>   +--1 (fused)
>Following the attempted fusion of parallel for-loops there are 3 parallel for-
>loop(s) (originating from loops labelled: #2, #3, #0).
>--------------------------------------------------------------------------------
>---------------------------- Optimising loop nests -----------------------------
>Attempting loop nest rewrites (optimising for the largest parallel loops)...
>
>+--3 is a parallel loop
>   +--0 --> rewritten as a serial loop
>--------------------------------------------------------------------------------
>----------------------------- Before Optimisation ------------------------------
>Parallel region 0:
>+--3 (parallel)
>   +--0 (parallel)
>   +--1 (parallel)
>
>
>--------------------------------------------------------------------------------
>------------------------------ After Optimisation ------------------------------
>Parallel region 0:
>+--3 (parallel)
>   +--0 (serial, fused with loop(s): 1)
>
>
>
>Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
> of the larger parallel loop (#3).
>--------------------------------------------------------------------------------
>--------------------------------------------------------------------------------
>
>---------------------------Loop invariant code motion---------------------------
>Allocation hoisting:
>The memory allocation derived from the instruction at 
>/Users/liwuchen/Library/Mobile 
>Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine 
>Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (180) is hoisted out
> of the parallel loop labelled #3 (it will be performed before the loop is 
>executed and reused inside the loop):
>   Allocation:: out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
>    - numpy.empty() is used for the allocation.
>The memory allocation derived from the instruction at 
>/Users/liwuchen/Library/Mobile 
>Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine 
>Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (181) is hoisted out
> of the parallel loop labelled #3 (it will be performed before the loop is 
>executed and reused inside the loop):
>   Allocation:: in_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
>    - numpy.empty() is used for the allocation.
>None
>ZIP
>
>================================================================================
> Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
>/Users/liwuchen/Library/Mobile 
>Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine 
>Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (216)  
>================================================================================
>
>
>Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/liwuchen/Library/Mobile Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (216) 
>-----------------------------------------------------------------------------|loop #ID
>    def _zip(                                                                | 
>        out: Storage,                                                        | 
>        out_shape: Shape,                                                    | 
>        out_strides: Strides,                                                | 
>        a_storage: Storage,                                                  | 
>        a_shape: Shape,                                                      | 
>        a_strides: Strides,                                                  | 
>        b_storage: Storage,                                                  | 
>        b_shape: Shape,                                                      | 
>        b_strides: Strides,                                                  | 
>    ) -> None:                                                               | 
>        stride_aligned = (                                                   | 
>            np.array_equal(out_strides, a_strides)                           | 
>            and np.array_equal(out_strides, b_strides)                       | 
>            and np.array_equal(out_shape, a_shape)                           | 
>            and np.array_equal(out_shape, b_shape)                           | 
>        )                                                                    | 
>        if stride_aligned:                                                   | 
>            for ordinal in prange(len(out)):---------------------------------| #7
>                # if strides are aligned, we can avoid indexing              | 
>                out[ordinal] = fn(a_storage[ordinal], b_storage[ordinal])    | 
>        else:                                                                | 
>            for ordinal in prange(len(out)):---------------------------------| #8
>                out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)--------| #4
>                a_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)----------| #5
>                b_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)----------| #6
>                # ordinal -> index                                           | 
>                to_index(ordinal, out_shape, out_index)                      | 
>                broadcast_index(out_index, out_shape, a_shape, a_index)      | 
>                broadcast_index(out_index, out_shape, b_shape, b_index)      | 
>                # index -> real ordinal in memory                            | 
>                o = index_to_position(out_index, out_strides)                | 
>                j = index_to_position(a_index, a_strides)                    | 
>                k = index_to_position(b_index, b_strides)                    | 
>                out[o] = fn(a_storage[j], b_storage[k])                      | 
>--------------------------------- Fusing loops ---------------------------------
>Attempting fusion of parallel loops (combines loops with similar properties)...
>
>Fused loop summary:
>+--4 has the following loops fused into it:
>   +--5 (fused)
>   +--6 (fused)
>Following the attempted fusion of parallel for-loops there are 3 parallel for-
>loop(s) (originating from loops labelled: #7, #8, #4).
>--------------------------------------------------------------------------------
>---------------------------- Optimising loop nests -----------------------------
>Attempting loop nest rewrites (optimising for the largest parallel loops)...
>
>+--8 is a parallel loop
>   +--4 --> rewritten as a serial loop
>--------------------------------------------------------------------------------
>----------------------------- Before Optimisation ------------------------------
>Parallel region 0:
>+--8 (parallel)
>   +--4 (parallel)
>   +--5 (parallel)
>   +--6 (parallel)
>
>
>--------------------------------------------------------------------------------
>------------------------------ After Optimisation ------------------------------
>Parallel region 0:
>+--8 (parallel)
>   +--4 (serial, fused with loop(s): 5, 6)
>
>
>
>Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part
> of the larger parallel loop (#8).
>--------------------------------------------------------------------------------
>--------------------------------------------------------------------------------
>
>---------------------------Loop invariant code motion---------------------------
>Allocation hoisting:
>The memory allocation derived from the instruction at 
>/Users/liwuchen/Library/Mobile 
>Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine 
>Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (239) is hoisted out
> of the parallel loop labelled #8 (it will be performed before the loop is 
>executed and reused inside the loop):
>   Allocation:: out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
>    - numpy.empty() is used for the allocation.
>The memory allocation derived from the instruction at 
>/Users/liwuchen/Library/Mobile 
>Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine 
>Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (240) is hoisted out
> of the parallel loop labelled #8 (it will be performed before the loop is 
>executed and reused inside the loop):
>   Allocation:: a_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
>    - numpy.empty() is used for the allocation.
>The memory allocation derived from the instruction at 
>/Users/liwuchen/Library/Mobile 
>Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine 
>Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (241) is hoisted out
> of the parallel loop labelled #8 (it will be performed before the loop is 
>executed and reused inside the loop):
>   Allocation:: b_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
>    - numpy.empty() is used for the allocation.
>None
>REDUCE
>
>================================================================================
> Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
>/Users/liwuchen/Library/Mobile 
>Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine 
>Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (276)  
>================================================================================
>
>
>Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/liwuchen/Library/Mobile Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (276) 
>---------------------------------------------------------------------|loop #ID
>    def _reduce(                                                     | 
>        out: Storage,                                                | 
>        out_shape: Shape,                                            | 
>        out_strides: Strides,                                        | 
>        a_storage: Storage,                                          | 
>        a_shape: Shape,                                              | 
>        a_strides: Strides,                                          | 
>        reduce_dim: int,                                             | 
>    ) -> None:                                                       | 
>        reduce_size = a_shape[reduce_dim]                            | 
>        # go through all index, starting from [0,0,0]                | 
>        for ordinal in prange(len(out)):-----------------------------| #10
>            out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)----| #9
>            # ordinal -> index                                       | 
>            to_index(ordinal, out_shape, out_index)                  | 
>            o = index_to_position(out_index, out_strides)            | 
>            temp = out[o]                                            | 
>            # reduce dimension                                       | 
>            for i in range(reduce_size):                             | 
>                # find the corresponding index in a                  | 
>                out_index[reduce_dim] = i                            | 
>                j = index_to_position(out_index, a_strides)          | 
>                temp = fn(temp, a_storage[j])                        | 
>            # write the result back to out                           | 
>            out[o] = temp                                            | 
>--------------------------------- Fusing loops ---------------------------------
>Attempting fusion of parallel loops (combines loops with similar properties)...
>Following the attempted fusion of parallel for-loops there are 2 parallel for-
>loop(s) (originating from loops labelled: #10, #9).
>--------------------------------------------------------------------------------
>---------------------------- Optimising loop nests -----------------------------
>Attempting loop nest rewrites (optimising for the largest parallel loops)...
>
>+--10 is a parallel loop
>   +--9 --> rewritten as a serial loop
>--------------------------------------------------------------------------------
>----------------------------- Before Optimisation ------------------------------
>Parallel region 0:
>+--10 (parallel)
>   +--9 (parallel)
>
>
>--------------------------------------------------------------------------------
>------------------------------ After Optimisation ------------------------------
>Parallel region 0:
>+--10 (parallel)
>   +--9 (serial)
>
>
>
>Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as 
>part of the larger parallel loop (#10).
>--------------------------------------------------------------------------------
>--------------------------------------------------------------------------------
>
>---------------------------Loop invariant code motion---------------------------
>Allocation hoisting:
>The memory allocation derived from the instruction at 
>/Users/liwuchen/Library/Mobile 
>Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine 
>Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (288) is hoisted out
> of the parallel loop labelled #10 (it will be performed before the loop is 
>executed and reused inside the loop):
>   Allocation:: out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
>    - numpy.empty() is used for the allocation.
>None
>MATRIX MULTIPLY
>
>================================================================================
> Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
>/Users/liwuchen/Library/Mobile 
>Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine 
>
>Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (305)  
>================================================================================
>
>
>Parallel loop listing for  Function _tensor_matrix_multiply, /Users/liwuchen/Library/Mobile Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (305) 
>---------------------------------------------------------------------------------------------------------|loop #ID
>def _tensor_matrix_multiply(                                                                             | 
>    out: Storage,                                                                                        | 
>    out_shape: Shape,                                                                                    | 
>    out_strides: Strides,                                                                                | 
>    a_storage: Storage,                                                                                  | 
>    a_shape: Shape,                                                                                      | 
>    a_strides: Strides,                                                                                  | 
>    b_storage: Storage,                                                                                  | 
>    b_shape: Shape,                                                                                      | 
>    b_strides: Strides,                                                                                  | 
>) -> None:                                                                                               | 
>    """NUMBA tensor matrix multiply function.                                                            | 
>                                                                                                         | 
>    Should work for any tensor shapes that broadcast as long as                                          | 
>                                                                                                         | 
>    ```                                                                                                  | 
>    assert a_shape[-1] == b_shape[-2]                                                                    | 
>    ```                                                                                                  | 
>                                                                                                         | 
>    Optimizations:                                                                                       | 
>                                                                                                         | 
>    * Outer loop in parallel                                                                             | 
>    * No index buffers or function calls                                                                 | 
>    * Inner loop should have no global writes, 1 multiply.                                               | 
>                                                                                                         | 
>                                                                                                         | 
>    Args:                                                                                                | 
>    ----                                                                                                 | 
>        out (Storage): storage for `out` tensor                                                          | 
>        out_shape (Shape): shape for `out` tensor                                                        | 
>        out_strides (Strides): strides for `out` tensor                                                  | 
>        a_storage (Storage): storage for `a` tensor                                                      | 
>        a_shape (Shape): shape for `a` tensor                                                            | 
>        a_strides (Strides): strides for `a` tensor                                                      | 
>        b_storage (Storage): storage for `b` tensor                                                      | 
>        b_shape (Shape): shape for `b` tensor                                                            | 
>        b_strides (Strides): strides for `b` tensor                                                      | 
>                                                                                                         | 
>    Returns:                                                                                             | 
>    -------                                                                                              | 
>        None : Fills in `out`                                                                            | 
>                                                                                                         | 
>    """                                                                                                  | 
>    # we assume a 3D matrix and the first dimension is the batch size                                    | 
>    assert a_shape[-1] == b_shape[-2]                                                                    | 
>    # get the column size                                                                                | 
>    column_size = a_shape[-1]                                                                            | 
>    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                               | 
>    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                               | 
>    for i in prange(out_shape[0]):-----------------------------------------------------------------------| #11
>        for j in range(out_shape[1]):                                                                    | 
>            for k in range(out_shape[2]):                                                                | 
>                temp = 0.0                                                                               | 
>                a_pos = i * a_batch_stride + j * a_strides[1]                                            | 
>                b_pos = i * b_batch_stride + k * b_strides[2]                                            | 
>                for n in range(column_size):                                                             | 
>                    temp += a_storage[a_pos + n * a_strides[2]] * b_storage[b_pos + n * b_strides[1]]    | 
>                out[i * out_strides[0] + j * out_strides[1] + k * out_strides[2]] = temp                 | 
>--------------------------------- Fusing loops ---------------------------------
>Attempting fusion of parallel loops (combines loops with similar properties)...
>Following the attempted fusion of parallel for-loops there are 1 parallel for-
>loop(s) (originating from loops labelled: #11).
>--------------------------------------------------------------------------------
>----------------------------- Before Optimisation ------------------------------
>--------------------------------------------------------------------------------
>------------------------------ After Optimisation ------------------------------
>Parallel structure is already optimal.
>--------------------------------------------------------------------------------
>--------------------------------------------------------------------------------
>
>---------------------------Loop invariant code motion---------------------------
>Allocation hoisting:
>No allocation hoisting found
>None