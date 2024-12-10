# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

# Diagnostics Output from parallel_check.py

```sh
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/liwuchen/Library/Mobile
Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine
Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (164)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/liwuchen/Library/Mobile Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (164)
----------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                           |
        out: Storage,                                                                   |
        out_shape: Shape,                                                               |
        out_strides: Strides,                                                           |
        in_storage: Storage,                                                            |
        in_shape: Shape,                                                                |
        in_strides: Strides,                                                            |
    ) -> None:                                                                          |
        stride_aligned = np.array_equal(out_strides, in_strides) and np.array_equal(    |
            out_shape, in_shape                                                         |
        )                                                                               |
        index: Index = np.zeros((2, len(out), MAX_DIMS), dtype=np.int32)----------------| #0
        if stride_aligned:                                                              |
            for ordinal in prange(len(out)):--------------------------------------------| #1
                # if strides are aligned, we can avoid indexing                         |
                out[ordinal] = fn(in_storage[ordinal])                                  |
        else:                                                                           |
            for ordinal in prange(len(out)):--------------------------------------------| #2
                out_index = index[0, ordinal]                                           |
                in_index = index[1, ordinal]                                            |
                # ordinal -> index                                                      |
                to_index(ordinal, out_shape, out_index)                                 |
                broadcast_index(out_index, out_shape, in_shape, in_index)               |
                # index -> real ordinal in memory                                       |
                o = index_to_position(out_index, out_strides)                           |
                j = index_to_position(in_index, in_strides)                             |
                out[o] = fn(in_storage[j])                                              |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #0, #1, #2).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/liwuchen/Library/Mobile
Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine
Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (218)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/liwuchen/Library/Mobile Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (218)
-----------------------------------------------------------------------------|loop #ID
    def _zip(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        a_storage: Storage,                                                  |
        a_shape: Shape,                                                      |
        a_strides: Strides,                                                  |
        b_storage: Storage,                                                  |
        b_shape: Shape,                                                      |
        b_strides: Strides,                                                  |
    ) -> None:                                                               |
        stride_aligned = (                                                   |
            np.array_equal(out_strides, a_strides)                           |
            and np.array_equal(out_strides, b_strides)                       |
            and np.array_equal(out_shape, a_shape)                           |
            and np.array_equal(out_shape, b_shape)                           |
        )                                                                    |
        index = np.zeros((3, len(out), MAX_DIMS), dtype=np.int32)------------| #3
        if stride_aligned:                                                   |
            for ordinal in prange(len(out)):---------------------------------| #4
                # if strides are aligned, we can avoid indexing              |
                out[ordinal] = fn(a_storage[ordinal], b_storage[ordinal])    |
        else:                                                                |
            for ordinal in prange(len(out)):---------------------------------| #5
                out_index = index[0, ordinal]                                |
                a_index = index[1, ordinal]                                  |
                b_index = index[2, ordinal]                                  |
                # ordinal -> index                                           |
                to_index(ordinal, out_shape, out_index)                      |
                broadcast_index(out_index, out_shape, a_shape, a_index)      |
                broadcast_index(out_index, out_shape, b_shape, b_index)      |
                # index -> real ordinal in memory                            |
                o = index_to_position(out_index, out_strides)                |
                j = index_to_position(a_index, a_strides)                    |
                k = index_to_position(b_index, b_strides)                    |
                out[o] = fn(a_storage[j], b_storage[k])                      |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #3, #4, #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/liwuchen/Library/Mobile
Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine
Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (279)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/liwuchen/Library/Mobile Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (279)
-------------------------------------------------------------------------|loop #ID
    def _reduce(                                                         |
        out: Storage,                                                    |
        out_shape: Shape,                                                |
        out_strides: Strides,                                            |
        a_storage: Storage,                                              |
        a_shape: Shape,                                                  |
        a_strides: Strides,                                              |
        reduce_dim: int,                                                 |
    ) -> None:                                                           |
        reduce_size = a_shape[reduce_dim]                                |
        # go through all index, starting from [0,0,0]                    |
        index: Index = np.zeros((len(out), MAX_DIMS), dtype=np.int32)----| #6
        for ordinal in prange(len(out)):---------------------------------| #7
            out_index = index[ordinal]                                   |
            # ordinal -> index                                           |
            to_index(ordinal, out_shape, out_index)                      |
            o = index_to_position(out_index, out_strides)                |
            temp = out[o]                                                |
            # reduce dimension                                           |
            for i in range(reduce_size):                                 |
                # find the corresponding index in a                      |
                out_index[reduce_dim] = i                                |
                j = index_to_position(out_index, a_strides)              |
                temp = fn(temp, a_storage[j])                            |
            # write the result back to out                               |
            out[o] = temp                                                |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #6, #7).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/liwuchen/Library/Mobile
Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine
Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (309)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/liwuchen/Library/Mobile Documents/com~apple~CloudDocs/Universities/Cornell/Courses/CS 5781 Machine Learning Engineering/mod3-SightVanish/minitorch/fast_ops.py (309)
--------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                |
    out: Storage,                                                                           |
    out_shape: Shape,                                                                       |
    out_strides: Strides,                                                                   |
    a_storage: Storage,                                                                     |
    a_shape: Shape,                                                                         |
    a_strides: Strides,                                                                     |
    b_storage: Storage,                                                                     |
    b_shape: Shape,                                                                         |
    b_strides: Strides,                                                                     |
) -> None:                                                                                  |
    """NUMBA tensor matrix multiply function.                                               |
                                                                                            |
    Should work for any tensor shapes that broadcast as long as                             |
                                                                                            |
    ```                                                                                     |
    assert a_shape[-1] == b_shape[-2]                                                       |
    ```                                                                                     |
                                                                                            |
    Optimizations:                                                                          |
                                                                                            |
    * Outer loop in parallel                                                                |
    * No index buffers or function calls                                                    |
    * Inner loop should have no global writes, 1 multiply.                                  |
                                                                                            |
                                                                                            |
    Args:                                                                                   |
    ----                                                                                    |
        out (Storage): storage for `out` tensor                                             |
        out_shape (Shape): shape for `out` tensor                                           |
        out_strides (Strides): strides for `out` tensor                                     |
        a_storage (Storage): storage for `a` tensor                                         |
        a_shape (Shape): shape for `a` tensor                                               |
        a_strides (Strides): strides for `a` tensor                                         |
        b_storage (Storage): storage for `b` tensor                                         |
        b_shape (Shape): shape for `b` tensor                                               |
        b_strides (Strides): strides for `b` tensor                                         |
                                                                                            |
    Returns:                                                                                |
    -------                                                                                 |
        None : Fills in `out`                                                               |
                                                                                            |
    """                                                                                     |
    # we assume a 3D matrix and the first dimension is the batch size                       |
    assert a_shape[-1] == b_shape[-2]                                                       |
    # get the column size                                                                   |
    column_size = a_shape[-1]                                                               |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                  |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                  |
    for i in prange(out_shape[0]):----------------------------------------------------------| #8
        for j in range(out_shape[1]):                                                       |
            for k in range(out_shape[2]):                                                   |
                temp = 0.0                                                                  |
                a_pos = i * a_batch_stride + j * a_strides[1]                               |
                b_pos = i * b_batch_stride + k * b_strides[2]                               |
                for n in range(column_size):                                                |
                    temp += (                                                               |
                        a_storage[a_pos + n * a_strides[2]]                                 |
                        * b_storage[b_pos + n * b_strides[1]]                               |
                    )                                                                       |
                out[i * out_strides[0] + j * out_strides[1] + k * out_strides[2]] = temp    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #8).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

# Task 3.4 CUDA vs. FastOps

```
Timing summary
Size: 64
    fast: 0.00421
    gpu: 0.00778
Size: 128
    fast: 0.01909
    gpu: 0.01496
Size: 256
    fast: 0.09510
    gpu: 0.05267
Size: 512
    fast: 1.12510
    gpu: 0.26931
Size: 1024
    fast: 7.81901
    gpu: 1.03830
```

# Task 3.5 Training

## Tensor Model CPU

### Split

`python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05`

>   Epoch 0, time 16.32s, loss 9.239515662299493, correct 24
>   Epoch 10, time 0.07s, loss 5.757932570155995, correct 42
>   Epoch 20, time 0.07s, loss 5.0081373800226165, correct 41
>   Epoch 30, time 0.07s, loss 4.174325925415107, correct 44
>   Epoch 40, time 0.07s, loss 2.9143587013718637, correct 48
>   Epoch 50, time 0.07s, loss 2.2453316140477613, correct 47
>   Epoch 60, time 0.07s, loss 2.4401030474486647, correct 48
>   Epoch 70, time 0.07s, loss 2.4906097867719508, correct 48
>   Epoch 80, time 0.07s, loss 2.011081351996485, correct 48
>   Epoch 90, time 0.07s, loss 0.999886456838797, correct 49
>   Epoch 100, time 0.07s, loss 0.5384038699254926, correct 49
>   Epoch 110, time 0.07s, loss 0.8234609667604398, correct 48
>   Epoch 120, time 0.07s, loss 0.6709149770741921, correct 49
>   Epoch 130, time 0.07s, loss 1.4883055793225866, correct 48
>   Epoch 140, time 0.07s, loss 1.503026675336136, correct 49
>   Epoch 150, time 0.07s, loss 0.9191848061769584, correct 49
>   Epoch 160, time 0.07s, loss 1.200076594284338, correct 49
>   Epoch 170, time 0.07s, loss 0.41069376064625485, correct 50
>   Epoch 180, time 0.07s, loss 1.5675215001108445, correct 49
>   Epoch 190, time 0.07s, loss 0.6558269121274413, correct 50
>   Epoch 200, time 0.07s, loss 0.4411292487059895, correct 50
>   Epoch 210, time 0.07s, loss 0.5759830207866089, correct 49
>   Epoch 220, time 0.07s, loss 0.4771679017360768, correct 50
>   Epoch 230, time 0.07s, loss 1.1677223437237798, correct 50
>   Epoch 240, time 0.07s, loss 0.2888341766632329, correct 50
>   Epoch 250, time 0.08s, loss 1.0136450518102182, correct 50
>   Epoch 260, time 0.07s, loss 0.8093947618384693, correct 49
>   Epoch 270, time 0.07s, loss 0.09833170667857272, correct 49
>   Epoch 280, time 0.07s, loss 1.1223567441043074, correct 50
>   Epoch 290, time 0.07s, loss 0.7225785117089514, correct 49
>   Epoch 300, time 0.07s, loss 0.16669165646414394, correct 50
>   Epoch 310, time 0.07s, loss 0.15768738127946236, correct 49
>   Epoch 320, time 0.07s, loss 0.02746715793882394, correct 50
>   Epoch 330, time 0.07s, loss 0.2777691382224904, correct 50
>   Epoch 340, time 0.07s, loss 0.10655855989105113, correct 49
>   Epoch 350, time 0.07s, loss 0.4156861983765441, correct 50
>   Epoch 360, time 0.07s, loss 0.027439907684297795, correct 50
>   Epoch 370, time 0.07s, loss 0.8857731271535647, correct 49
>   Epoch 380, time 0.07s, loss 0.8119048015826751, correct 50
>   Epoch 390, time 0.07s, loss 0.5753970338124791, correct 50
>   Epoch 400, time 0.07s, loss 0.03686510754304407, correct 50
>   Epoch 410, time 0.07s, loss 0.867685114918241, correct 50
>   Epoch 420, time 0.07s, loss 0.18723282764561455, correct 49
>   Epoch 430, time 0.07s, loss 0.0983647481056742, correct 49
>   Epoch 440, time 0.07s, loss 0.11701785386381744, correct 50
>   Epoch 450, time 0.07s, loss 0.0828865951812304, correct 49
>   Epoch 460, time 0.07s, loss 0.08753602765523993, correct 50
>   Epoch 470, time 0.07s, loss 0.06481900165345678, correct 50
>   Epoch 480, time 0.07s, loss 0.5627684315242953, correct 50
>   Epoch 490, time 0.07s, loss 0.09082600432944322, correct 50

### Simple

`python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.
05`

>   Epoch 0, time 16.04s, loss 4.631628926807737, correct 48
>   Epoch 10, time 0.07s, loss 1.9444326385319148, correct 49
>   Epoch 20, time 0.07s, loss 0.987595404382914, correct 49
>   Epoch 30, time 0.07s, loss 0.642656249960495, correct 49
>   Epoch 40, time 0.07s, loss 1.0816096087740164, correct 50
>   Epoch 50, time 0.07s, loss 1.2923101902988903, correct 50
>   Epoch 60, time 0.07s, loss 0.877594161863142, correct 50
>   Epoch 70, time 0.07s, loss 0.052043764648277424, correct 50
>   Epoch 80, time 0.07s, loss 0.7535313367525789, correct 50
>   Epoch 90, time 0.07s, loss 0.31571679145946957, correct 50
>   Epoch 100, time 0.07s, loss 0.38947630727537175, correct 50
>   Epoch 110, time 0.07s, loss 0.09069226716282769, correct 50
>   Epoch 120, time 0.07s, loss 0.06393477807005596, correct 50
>   Epoch 130, time 0.07s, loss 0.05290480111906398, correct 50
>   Epoch 140, time 0.07s, loss 0.11655537831990959, correct 50
>   Epoch 150, time 0.07s, loss 0.39821289023056367, correct 50
>   Epoch 160, time 0.07s, loss 0.20592136136723754, correct 50
>   Epoch 170, time 0.07s, loss 0.27629504019319456, correct 50
>   Epoch 180, time 0.07s, loss 0.28041377363312664, correct 50
>   Epoch 190, time 0.07s, loss 0.3159381976349352, correct 50
>   Epoch 200, time 0.07s, loss 0.03771306538119334, correct 50
>   Epoch 210, time 0.07s, loss 0.6619092768262431, correct 50
>   Epoch 220, time 0.07s, loss 0.0845296821425622, correct 50
>   Epoch 230, time 0.07s, loss 0.03472367035579978, correct 50
>   Epoch 240, time 0.07s, loss 0.185706404592004, correct 50
>   Epoch 250, time 0.07s, loss 0.04769916324742647, correct 50
>   Epoch 260, time 0.07s, loss 0.33647624897556117, correct 50
>   Epoch 270, time 0.07s, loss 0.25361958159697884, correct 50
>   Epoch 280, time 0.07s, loss 0.003230115363709347, correct 50
>   Epoch 290, time 0.07s, loss 0.24647090775253339, correct 50
>   Epoch 300, time 0.07s, loss 0.07525472879382158, correct 50
>   Epoch 310, time 0.07s, loss 0.2454483310983242, correct 50
>   Epoch 320, time 0.07s, loss 0.27515040428078136, correct 50
>   Epoch 330, time 0.07s, loss 0.02696523642345113, correct 50
>   Epoch 340, time 0.07s, loss 0.03745942507657725, correct 50
>   Epoch 350, time 0.07s, loss 0.19265684212327477, correct 50
>   Epoch 360, time 0.07s, loss 0.21058436401435662, correct 50
>   Epoch 370, time 0.07s, loss 0.1253761177657817, correct 50
>   Epoch 380, time 0.07s, loss 0.00010516500345630241, correct 50
>   Epoch 390, time 0.07s, loss 0.03656088362928413, correct 50
>   Epoch 400, time 0.07s, loss 0.18996003367810563, correct 50
>   Epoch 410, time 0.07s, loss 0.18759559409792015, correct 50
>   Epoch 420, time 0.07s, loss 0.01947562815578315, correct 50
>   Epoch 430, time 0.07s, loss 0.1888340301395624, correct 50
>   Epoch 440, time 0.07s, loss 0.09857984998695453, correct 50
>   Epoch 450, time 0.07s, loss 0.18669065678501784, correct 50
>   Epoch 460, time 0.07s, loss 0.008090928942390961, correct 50
>   Epoch 470, time 0.07s, loss 0.06421158341633892, correct 50
>   Epoch 480, time 0.07s, loss 0.0008610554575453309, correct 50
>   Epoch 490, time 0.07s, loss 0.1506280914001564, correct 50

### Xor

`python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05`

>   Epoch 0, time 16.18s, loss 7.344124042183289, correct 17
>   Epoch 10, time 0.07s, loss 4.0758451817942944, correct 38
>   Epoch 20, time 0.07s, loss 4.114100535604485, correct 47
>   Epoch 30, time 0.07s, loss 2.3939549625440595, correct 47
>   Epoch 40, time 0.07s, loss 5.47503288364324, correct 47
>   Epoch 50, time 0.07s, loss 2.540341177470371, correct 47
>   Epoch 60, time 0.07s, loss 1.6502847072178006, correct 49
>   Epoch 70, time 0.07s, loss 2.823714325990196, correct 47
>   Epoch 80, time 0.07s, loss 3.7655858721501305, correct 49
>   Epoch 90, time 0.07s, loss 0.7215903736276068, correct 49
>   Epoch 100, time 0.07s, loss 1.4794977635688293, correct 49
>   Epoch 110, time 0.07s, loss 2.8230948450308504, correct 47
>   Epoch 120, time 0.07s, loss 1.287782115583424, correct 50
>   Epoch 130, time 0.07s, loss 1.3684247080636138, correct 48
>   Epoch 140, time 0.07s, loss 0.7744462246528383, correct 50
>   Epoch 150, time 0.07s, loss 1.2875752869473458, correct 50
>   Epoch 160, time 0.07s, loss 0.5842846654390436, correct 50
>   Epoch 170, time 0.07s, loss 0.36427458223008763, correct 49
>   Epoch 180, time 0.07s, loss 1.2479720889476968, correct 50
>   Epoch 190, time 0.07s, loss 0.6542936533696627, correct 49
>   Epoch 200, time 0.07s, loss 0.37904149833691086, correct 50
>   Epoch 210, time 0.07s, loss 0.8926802542438788, correct 50
>   Epoch 220, time 0.07s, loss 0.8103752323591296, correct 50
>   Epoch 230, time 0.07s, loss 1.3868856413856685, correct 50
>   Epoch 240, time 0.07s, loss 0.9276977201093679, correct 50
>   Epoch 250, time 0.08s, loss 0.9805064543219425, correct 50
>   Epoch 260, time 0.07s, loss 0.585012015757014, correct 50
>   Epoch 270, time 0.07s, loss 0.340226501224252, correct 50
>   Epoch 280, time 0.07s, loss 0.7152023777819674, correct 50
>   Epoch 290, time 0.07s, loss 0.4832025865134533, correct 50
>   Epoch 300, time 0.07s, loss 1.6490923792080245, correct 50
>   Epoch 310, time 0.07s, loss 0.44558254015394605, correct 50
>   Epoch 320, time 0.07s, loss 0.25370264621618427, correct 50
>   Epoch 330, time 0.07s, loss 0.6198355856499319, correct 50
>   Epoch 340, time 0.07s, loss 0.2505919861704662, correct 50
>   Epoch 350, time 0.07s, loss 0.49811245384700864, correct 50
>   Epoch 360, time 0.07s, loss 0.5824578155826353, correct 50
>   Epoch 370, time 0.07s, loss 0.056578676755452396, correct 50
>   Epoch 380, time 0.07s, loss 0.1822309590076544, correct 50
>   Epoch 390, time 0.07s, loss 0.2945572624912563, correct 50
>   Epoch 400, time 0.07s, loss 0.5312470343826448, correct 50
>   Epoch 410, time 0.07s, loss 0.40393236930734566, correct 50
>   Epoch 420, time 0.07s, loss 0.3108449850246913, correct 50
>   Epoch 430, time 0.07s, loss 0.13506976948297844, correct 50
>   Epoch 440, time 0.07s, loss 0.16684678227048064, correct 50
>   Epoch 450, time 0.07s, loss 0.05299663577939567, correct 50
>   Epoch 460, time 0.07s, loss 0.5355271427685412, correct 50
>   Epoch 470, time 0.07s, loss 0.07804866713016884, correct 50
>   Epoch 480, time 0.07s, loss 0.3998576932595814, correct 50
>   Epoch 490, time 0.07s, loss 0.25292656727483803, correct 50

## Tensor Model GPU

### Split

`run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.0
5`

>   Epoch 0, time 4.75s, loss 5.672913746386637, correct 33
>   Epoch 10, time 1.40s, loss 4.488143717816341, correct 34
>   Epoch 20, time 1.48s, loss 5.1330965494702605, correct 38
>   Epoch 30, time 1.38s, loss 5.557221681587952, correct 39
>   Epoch 40, time 1.41s, loss 4.3843604132500404, correct 45
>   Epoch 50, time 1.48s, loss 3.338512124941616, correct 36
>   Epoch 60, time 1.38s, loss 5.319105584121331, correct 43
>   Epoch 70, time 1.37s, loss 2.4884598443915977, correct 39
>   Epoch 80, time 1.41s, loss 1.9475908934077268, correct 39
>   Epoch 90, time 1.37s, loss 3.5011271715431986, correct 49
>   Epoch 100, time 1.36s, loss 3.2914949643961213, correct 48
>   Epoch 110, time 1.40s, loss 2.8929974570001917, correct 43
>   Epoch 120, time 1.38s, loss 2.0356338067098068, correct 48
>   Epoch 130, time 1.45s, loss 3.8444202124243367, correct 47
>   Epoch 140, time 1.38s, loss 1.962562233966721, correct 48
>   Epoch 150, time 1.39s, loss 0.6570472391614002, correct 47
>   Epoch 160, time 1.46s, loss 1.9014480997920347, correct 49
>   Epoch 170, time 1.38s, loss 0.5986356018973511, correct 49
>   Epoch 180, time 1.38s, loss 2.8506625542047694, correct 48
>   Epoch 190, time 1.44s, loss 1.7883785282107152, correct 49
>   Epoch 200, time 1.37s, loss 1.754586118360078, correct 49
>   Epoch 210, time 1.38s, loss 0.8582607054127747, correct 50
>   Epoch 220, time 1.45s, loss 2.5563234278244034, correct 45
>   Epoch 230, time 1.40s, loss 0.5662426394367909, correct 49
>   Epoch 240, time 1.38s, loss 0.51524089405967, correct 48
>   Epoch 250, time 1.43s, loss 2.6115049430615214, correct 50
>   Epoch 260, time 1.39s, loss 0.7984639803657767, correct 48
>   Epoch 270, time 1.40s, loss 0.5652921154417563, correct 49
>   Epoch 280, time 1.40s, loss 1.3242416792696035, correct 49
>   Epoch 290, time 1.38s, loss 2.55090197837839, correct 41
>   Epoch 300, time 1.39s, loss 0.5607277732285696, correct 48
>   Epoch 310, time 1.37s, loss 1.57317392711006, correct 50
>   Epoch 320, time 1.38s, loss 0.6041443172804896, correct 49
>   Epoch 330, time 1.38s, loss 0.4542139544917243, correct 50
>   Epoch 340, time 1.45s, loss 0.5034425160067358, correct 48
>   Epoch 350, time 1.38s, loss 0.09760444177920485, correct 49
>   Epoch 360, time 1.38s, loss 1.4806007371782204, correct 50
>   Epoch 370, time 1.38s, loss 0.8185317880036238, correct 50
>   Epoch 380, time 1.61s, loss 0.24273928437678177, correct 49
>   Epoch 390, time 1.38s, loss 0.02804714556189953, correct 48
>   Epoch 400, time 1.37s, loss 0.4833815054621354, correct 49
>   Epoch 410, time 1.38s, loss 0.7660882714602748, correct 49
>   Epoch 420, time 1.43s, loss 0.3537809037315294, correct 50
>   Epoch 430, time 1.38s, loss 0.39258714874149114, correct 50
>   Epoch 440, time 1.38s, loss 0.12132652080034374, correct 50
>   Epoch 450, time 1.45s, loss 0.025458604751605007, correct 49
>   Epoch 460, time 1.43s, loss 0.9360907713745789, correct 49
>   Epoch 470, time 1.38s, loss 0.8416508557049092, correct 50
>   Epoch 480, time 1.46s, loss 0.4877653194869489, correct 50
>   Epoch 490, time 1.39s, loss 0.7929105451872799, correct 50

### Simple

`python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05`

>   Epoch 0, time 4.73s, loss 6.289625391653283, correct 35
>   Epoch 10, time 1.41s, loss 1.344562294671013, correct 46
>   Epoch 20, time 1.50s, loss 1.2252637382991618, correct 50
>   Epoch 30, time 1.41s, loss 2.5826081394034723, correct 46
>   Epoch 40, time 1.40s, loss 1.339657531419103, correct 50
>   Epoch 50, time 1.47s, loss 1.071085161777824, correct 50
>   Epoch 60, time 1.40s, loss 0.5367286991707202, correct 50
>   Epoch 70, time 1.41s, loss 0.7164098973598503, correct 50
>   Epoch 80, time 1.40s, loss 0.025051570831173202, correct 50
>   Epoch 90, time 1.41s, loss 1.1823255264694907, correct 50
>   Epoch 100, time 1.42s, loss 0.7723203255754701, correct 50
>   Epoch 110, time 1.42s, loss 0.07456817333148151, correct 50
>   Epoch 120, time 1.42s, loss 0.9008050231568288, correct 50
>   Epoch 130, time 1.46s, loss 0.4594980841796705, correct 50
>   Epoch 140, time 1.40s, loss 0.03358691195832393, correct 50
>   Epoch 150, time 1.40s, loss 0.457337749862607, correct 50
>   Epoch 160, time 1.49s, loss 0.8429036565731518, correct 50
>   Epoch 170, time 1.40s, loss 0.20275358493808837, correct 50
>   Epoch 180, time 1.41s, loss 0.19171439071846452, correct 50
>   Epoch 190, time 1.48s, loss 0.6157415755433243, correct 50
>   Epoch 200, time 1.41s, loss 0.30745929974596664, correct 50
>   Epoch 210, time 1.40s, loss 0.1420962932386814, correct 50
>   Epoch 220, time 1.48s, loss 0.19772587292768012, correct 50
>   Epoch 230, time 1.41s, loss 0.3337019450677819, correct 50
>   Epoch 240, time 1.42s, loss 0.8340860934483583, correct 50
>   Epoch 250, time 1.49s, loss 0.011285192559314627, correct 50
>   Epoch 260, time 1.41s, loss 0.4635105170301056, correct 50
>   Epoch 270, time 1.40s, loss 0.20692597585524483, correct 50
>   Epoch 280, time 1.40s, loss 0.5909122946471944, correct 50
>   Epoch 290, time 1.41s, loss 0.9503716897826021, correct 50
>   Epoch 300, time 1.41s, loss 0.61032414592484, correct 50
>   Epoch 310, time 1.42s, loss 0.4087213342284328, correct 50
>   Epoch 320, time 1.41s, loss 0.19477146715736704, correct 50
>   Epoch 330, time 1.41s, loss 0.11933660334486569, correct 50
>   Epoch 340, time 1.47s, loss 0.5784805132885126, correct 50
>   Epoch 350, time 1.41s, loss 0.18033728719489117, correct 50
>   Epoch 360, time 1.41s, loss 0.3771095912518696, correct 50
>   Epoch 370, time 1.42s, loss 0.214653104137022, correct 50
>   Epoch 380, time 1.42s, loss 0.4097716776393199, correct 50
>   Epoch 390, time 1.42s, loss 0.3235562206535017, correct 50
>   Epoch 400, time 1.41s, loss 0.12628054175257705, correct 50
>   Epoch 410, time 1.42s, loss 0.6960300709744667, correct 50
>   Epoch 420, time 1.47s, loss 0.05245172759676535, correct 50
>   Epoch 430, time 1.41s, loss 0.038785199229483334, correct 50
>   Epoch 440, time 1.40s, loss 0.40249578747786663, correct 50
>   Epoch 450, time 1.48s, loss 2.659628405556062e-05, correct 50
>   Epoch 460, time 1.42s, loss -2.594495907476261e-07, correct 50
>   Epoch 470, time 1.42s, loss 0.0018026202609719387, correct 50
>   Epoch 480, time 1.47s, loss 0.11063428360935491, correct 50
>   Epoch 490, time 1.41s, loss 0.00027558296011427527, correct 50

### Xor

`python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05`

>Epoch 0, time 4.73s, loss 6.931189260752635, correct 36
>Epoch 10, time 1.38s, loss 4.069956148445998, correct 41
>Epoch 20, time 1.44s, loss 3.1447807882772336, correct 48
>Epoch 30, time 1.38s, loss 2.582869997024142, correct 47
>Epoch 40, time 1.38s, loss 2.487593702122413, correct 49
>Epoch 50, time 1.46s, loss 1.827444921094747, correct 49
>Epoch 60, time 1.38s, loss 2.0140588770754566, correct 49
>Epoch 70, time 1.38s, loss 0.8716358379126269, correct 49
>Epoch 80, time 1.38s, loss 1.183558530076425, correct 49
>Epoch 90, time 1.36s, loss 1.419266236573993, correct 49
>Epoch 100, time 1.37s, loss 1.2601514458222316, correct 49
>Epoch 110, time 1.38s, loss 0.5906767040531897, correct 49
>Epoch 120, time 1.38s, loss 0.5425490706419178, correct 49
>Epoch 130, time 1.42s, loss 0.45221105950233437, correct 49
>Epoch 140, time 1.37s, loss 0.5862723772393749, correct 49
>Epoch 150, time 1.37s, loss 0.9032587615609872, correct 49
>Epoch 160, time 1.43s, loss 0.7853844582632612, correct 49
>Epoch 170, time 1.36s, loss 0.33375219300257514, correct 49
>Epoch 180, time 1.36s, loss 1.7447965139349837, correct 50
>Epoch 190, time 1.46s, loss 0.6585669278563993, correct 49
>Epoch 200, time 1.37s, loss 1.641104773947318, correct 50
>Epoch 210, time 1.39s, loss 0.7754418431291963, correct 49
>Epoch 220, time 1.43s, loss 0.22448844544822483, correct 49
>Epoch 230, time 1.37s, loss 0.13606656326283964, correct 49
>Epoch 240, time 1.36s, loss 0.4159494779764735, correct 50
>Epoch 250, time 1.43s, loss 1.4392506510266487, correct 50
>Epoch 260, time 1.37s, loss 0.5102129474875959, correct 49
>Epoch 270, time 1.37s, loss 0.3508316295367913, correct 49
>Epoch 280, time 1.37s, loss 0.19413046526894492, correct 49
>Epoch 290, time 1.38s, loss 0.4986517780794688, correct 49
>Epoch 300, time 1.38s, loss 0.2300910471239905, correct 49
>Epoch 310, time 1.38s, loss 0.358597110746053, correct 49
>Epoch 320, time 1.37s, loss 1.071161905936446, correct 50
>Epoch 330, time 1.38s, loss 0.4741450955451096, correct 49
>Epoch 340, time 1.47s, loss 0.9241771290700243, correct 50
>Epoch 350, time 1.38s, loss 0.16292083694238446, correct 49
>Epoch 360, time 1.37s, loss 0.1897407268830619, correct 50
>Epoch 370, time 1.37s, loss 1.111528551941698, correct 50
>Epoch 380, time 1.37s, loss 0.16598307974867837, correct 50
>Epoch 390, time 1.37s, loss 0.053111718848268694, correct 49
>Epoch 400, time 1.37s, loss 0.14638724823060636, correct 50
>Epoch 410, time 1.37s, loss 0.12516860781486008, correct 50
>Epoch 420, time 1.43s, loss 0.7178136288368929, correct 50
>Epoch 430, time 1.36s, loss 0.8793332173143742, correct 50
>Epoch 440, time 1.37s, loss 1.174097074818129, correct 50
>Epoch 450, time 1.45s, loss 0.7643722273833599, correct 50
>Epoch 460, time 1.37s, loss 0.2340838936446676, correct 50
>Epoch 470, time 1.36s, loss 0.3371617722773287, correct 50
>Epoch 480, time 1.47s, loss 0.14593870478320334, correct 50
>Epoch 490, time 1.38s, loss 0.9317017042769198, correct 50

## Big Model CPU

`python project/run_fast_tensor.py --BACKEND cp
u --HIDDEN 200 --DATASET simple --RATE 0.05`

>Epoch 0, time 16.40s, loss 4.352062991716662, correct 48
>Epoch 10, time 0.17s, loss 0.5944444319699129, correct 50
>Epoch 20, time 0.17s, loss 1.191825298072593, correct 50
>Epoch 30, time 0.17s, loss 0.4121081548313936, correct 50
>Epoch 40, time 0.17s, loss 0.19939063391397413, correct 50
>Epoch 50, time 0.17s, loss 0.20020630412676144, correct 50
>Epoch 60, time 0.17s, loss 0.30216094864316645, correct 50
>Epoch 70, time 0.17s, loss 0.27243098940199695, correct 50
>Epoch 80, time 0.17s, loss 0.6318471303340428, correct 50
>Epoch 90, time 0.17s, loss 0.7146857600797153, correct 50
>Epoch 100, time 0.17s, loss 0.6540349487731432, correct 50
>Epoch 110, time 0.17s, loss 0.09057362117995842, correct 50
>Epoch 120, time 0.17s, loss 0.1394931194057981, correct 50
>Epoch 130, time 0.17s, loss 0.0835887975276666, correct 50
>Epoch 140, time 0.17s, loss 0.40489535184329656, correct 50
>Epoch 150, time 0.17s, loss 0.05855542670479369, correct 50
>Epoch 160, time 0.17s, loss 0.005008244096881302, correct 50
>Epoch 170, time 0.17s, loss 0.01958915410872481, correct 50
>Epoch 180, time 0.17s, loss 0.08228627133554828, correct 50
>Epoch 190, time 0.17s, loss 0.11273464682690594, correct 50
>Epoch 200, time 0.17s, loss 0.2885563425336434, correct 50
>Epoch 210, time 0.17s, loss 0.009057650848848565, correct 50
>Epoch 220, time 0.17s, loss 0.14749082637664027, correct 50
>Epoch 230, time 0.17s, loss 0.18903295279966603, correct 50
>Epoch 240, time 0.17s, loss 0.25258955029112184, correct 50
>Epoch 250, time 0.17s, loss 0.361063605264579, correct 50
>Epoch 260, time 0.17s, loss 0.10465927723167706, correct 50
>Epoch 270, time 0.17s, loss 0.08186428412738417, correct 50
>Epoch 280, time 0.17s, loss 0.01208513233571684, correct 50
>Epoch 290, time 0.17s, loss 0.017736345542816598, correct 50
>Epoch 300, time 0.17s, loss 0.09209390842040566, correct 50
>Epoch 310, time 0.17s, loss 0.1472696214190044, correct 50
>Epoch 320, time 0.17s, loss 0.04180056329871825, correct 50
>Epoch 330, time 0.17s, loss 0.203010019938457, correct 50
>Epoch 340, time 0.17s, loss 0.008289873225194321, correct 50
>Epoch 350, time 0.17s, loss 0.006374458718372489, correct 50
>Epoch 360, time 0.17s, loss 0.11111017591879982, correct 50
>Epoch 370, time 0.17s, loss 0.20610353889144853, correct 50
>Epoch 380, time 0.17s, loss 0.0005129282097606368, correct 50
>Epoch 390, time 0.17s, loss 0.21098013661591422, correct 50
>Epoch 400, time 0.17s, loss 0.02733655564156088, correct 50
>Epoch 410, time 0.17s, loss 0.0460140700106841, correct 50
>Epoch 420, time 0.17s, loss 0.08944925660241322, correct 50
>Epoch 430, time 0.17s, loss 0.028024390816405183, correct 50
>Epoch 440, time 0.17s, loss 0.001204096286771791, correct 50
>Epoch 450, time 0.17s, loss 0.05315131921942401, correct 50
>Epoch 460, time 0.17s, loss 0.12987260254329294, correct 50
>Epoch 470, time 0.17s, loss 0.00506069339936265, correct 50
>Epoch 480, time 0.17s, loss 0.13891655495487443, correct 50
>Epoch 490, time 0.17s, loss 0.008952207541350845, correct 50

## Big Model GPU

`python project/run_fast_tensor.py --BACKEND gp
u --HIDDEN 200 --DATASET simple --RATE 0.05`

>   Epoch 0, time 4.87s, loss 5.319674836227793, correct 37
>   Epoch 10, time 1.48s, loss 1.9798629708380793, correct 45
>   Epoch 20, time 1.61s, loss 0.49790967632462274, correct 46
>   Epoch 30, time 1.52s, loss 2.3121117109097264, correct 45
>   Epoch 40, time 1.47s, loss 0.7374865790222596, correct 50
>   Epoch 50, time 1.54s, loss 0.665973446934982, correct 50
>   Epoch 60, time 1.47s, loss 0.44373766293375955, correct 50
>   Epoch 70, time 1.49s, loss 0.7154246042178237, correct 49
>   Epoch 80, time 1.47s, loss 0.7606424346163907, correct 50
>   Epoch 90, time 1.49s, loss 0.4210800132409975, correct 48
>   Epoch 100, time 1.48s, loss 0.007278754126664001, correct 47
>   Epoch 110, time 1.49s, loss 1.7079431858034217, correct 47
>   Epoch 120, time 1.48s, loss 0.8312124534140163, correct 50
>   Epoch 130, time 1.53s, loss 1.1432961001554034, correct 49
>   Epoch 140, time 1.48s, loss 0.7115027044285203, correct 50
>   Epoch 150, time 1.53s, loss 0.9627696958016001, correct 50
>   Epoch 160, time 1.52s, loss 1.353692856638173, correct 49
>   Epoch 170, time 1.48s, loss 1.0060371699382356, correct 50
>   Epoch 180, time 1.47s, loss 0.5522175839103525, correct 50
>   Epoch 190, time 1.53s, loss 0.4211482561627769, correct 50
>   Epoch 200, time 1.47s, loss 0.24410504499145722, correct 50
>   Epoch 210, time 1.47s, loss 0.334693484256275, correct 50
>   Epoch 220, time 1.57s, loss 0.17764184167959182, correct 50
>   Epoch 230, time 1.46s, loss 0.19888378669630202, correct 50
>   Epoch 240, time 1.47s, loss 0.7392897942467168, correct 50
>   Epoch 250, time 1.56s, loss 0.0016362429757807964, correct 50
>   Epoch 260, time 1.48s, loss 0.5428665367828571, correct 50
>   Epoch 270, time 1.49s, loss 0.5817308450787143, correct 50
>   Epoch 280, time 1.48s, loss 2.527705412716156e-05, correct 50
>   Epoch 290, time 1.49s, loss 0.2774585326150709, correct 50
>   Epoch 300, time 1.51s, loss 0.020643434035327705, correct 50
>   Epoch 310, time 1.52s, loss 0.08866414465176575, correct 50
>   Epoch 320, time 1.47s, loss 0.20840975213864177, correct 50
>   Epoch 330, time 1.49s, loss 0.01670286739863104, correct 50
>   Epoch 340, time 1.55s, loss 0.23582254422636376, correct 50
>   Epoch 350, time 1.49s, loss 0.159179700756148, correct 50
>   Epoch 360, time 1.47s, loss 0.5436308108073942, correct 50
>   Epoch 370, time 1.48s, loss 0.04432809351372472, correct 50
>   Epoch 380, time 1.49s, loss 0.12889905701170135, correct 50
>   Epoch 390, time 1.48s, loss 0.43099338901312034, correct 50
>   Epoch 400, time 1.49s, loss 0.02204235368673479, correct 50
>   Epoch 410, time 1.47s, loss 0.27339245954269586, correct 50
>   Epoch 420, time 1.54s, loss 0.0034527749562907246, correct 50
>   Epoch 430, time 1.48s, loss 0.4734279427201689, correct 50
>   Epoch 440, time 1.48s, loss 0.3943184043447154, correct 50
>   Epoch 450, time 1.57s, loss 0.005939516606727469, correct 50
>   Epoch 460, time 1.50s, loss 0.051978970106411296, correct 50
>   Epoch 470, time 1.50s, loss 0.20740024869836737, correct 50
>   Epoch 480, time 1.54s, loss -3.819844058665908e-06, correct 50
>   Epoch 490, time 1.49s, loss 0.08670240915976654, correct 50