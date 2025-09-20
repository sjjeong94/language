"""
Example vector kernels for testing vector operations.

These examples demonstrate various vector operation patterns
that should compile correctly to MLIR.
"""

import oven.language as ol


def simple_vector_copy(input_ptr: ol.ptr, output_ptr: ol.ptr):
    """Simple 1:1 vector copy operation."""
    tid = ol.get_tid_x()
    offset = tid * 4

    vector_data = ol.vload(input_ptr, offset, 4)
    ol.vstore(vector_data, output_ptr, offset, 4)


def vector_element_wise_add(input1_ptr: ol.ptr, input2_ptr: ol.ptr, output_ptr: ol.ptr):
    """Element-wise vector addition."""
    tid = ol.get_tid_x()
    bid = ol.get_bid_x()
    block_size = ol.get_bdim_x()

    global_idx = bid * block_size + tid
    offset = global_idx * 4

    vec1 = ol.vload(input1_ptr, offset, 4)
    vec2 = ol.vload(input2_ptr, offset, 4)
    result = vec1 + vec2
    ol.vstore(result, output_ptr, offset, 4)


def vector_mathematical_pipeline(input_ptr: ol.ptr, output_ptr: ol.ptr):
    """Chained mathematical operations on vectors."""
    tid = ol.get_tid_x()
    bid = ol.get_bid_x()
    block_size = ol.get_bdim_x()

    global_idx = bid * block_size + tid
    offset = global_idx * 4

    # Load vector
    vector_data = ol.vload(input_ptr, offset, 4)

    # Apply mathematical transformations
    vector_data = ol.sigmoid(vector_data)
    vector_data = ol.exp(vector_data)

    # Store result
    ol.vstore(vector_data, output_ptr, offset, 4)


def vector_mixed_operations(input_ptr: ol.ptr, output_ptr: ol.ptr, scale_factor: float):
    """Mixed scalar and vector operations."""
    tid = ol.get_tid_x()
    bid = ol.get_bid_x()
    block_size = ol.get_bdim_x()

    # Scalar computations
    global_idx = bid * block_size + tid
    offset = global_idx * 4

    # Vector operations
    vector_data = ol.vload(input_ptr, offset, 4)

    # Apply transformations
    transformed = ol.sigmoid(vector_data)

    # Store result
    ol.vstore(transformed, output_ptr, offset, 4)


def vector_size_variants(input_ptr: ol.ptr, output_ptr: ol.ptr):
    """Demonstrate different vector sizes."""
    tid = ol.get_tid_x()

    # 2-element vectors
    offset_2 = tid * 2
    vec2 = ol.vload(input_ptr, offset_2, 2)
    ol.vstore(vec2, output_ptr, offset_2, 2)

    # 4-element vectors
    offset_4 = tid * 4
    vec4 = ol.vload(input_ptr, offset_4, 4)
    ol.vstore(vec4, output_ptr, offset_4, 4)

    # 8-element vectors
    offset_8 = tid * 8
    vec8 = ol.vload(input_ptr, offset_8, 8)
    ol.vstore(vec8, output_ptr, offset_8, 8)


def vector_complex_arithmetic(
    ptr1: ol.ptr, ptr2: ol.ptr, ptr3: ol.ptr, output_ptr: ol.ptr
):
    """Complex arithmetic with multiple vectors."""
    tid = ol.get_tid_x()
    offset = tid * 4

    # Load three vectors
    vec1 = ol.vload(ptr1, offset, 4)
    vec2 = ol.vload(ptr2, offset, 4)
    vec3 = ol.vload(ptr3, offset, 4)

    # Complex expression: (vec1 + vec2) * vec3
    temp = vec1 + vec2
    result = temp * vec3

    # Store result
    ol.vstore(result, output_ptr, offset, 4)


def vector_conditional_processing(input_ptr: ol.ptr, output_ptr: ol.ptr, n: int):
    """Vector operations with conditional execution."""
    tid = ol.get_tid_x()
    bid = ol.get_bid_x()
    block_size = ol.get_bdim_x()

    global_idx = bid * block_size + tid

    # Bounds checking for vector operations
    if global_idx * 4 < n:
        offset = global_idx * 4

        # Safe to perform vector operation
        vector_data = ol.vload(input_ptr, offset, 4)
        processed = ol.exp(vector_data)
        ol.vstore(processed, output_ptr, offset, 4)


def vector_reduction_prep(input_ptr: ol.ptr, temp_ptr: ol.ptr):
    """Prepare vectors for reduction operations."""
    tid = ol.get_tid_x()
    bid = ol.get_bid_x()
    block_size = ol.get_bdim_x()

    global_idx = bid * block_size + tid
    offset = global_idx * 4

    # Load and preprocess vectors
    vector_data = ol.vload(input_ptr, offset, 4)

    # Apply preprocessing (e.g., absolute value approximation)
    processed = ol.sigmoid(vector_data)  # Normalize to [0,1]

    # Store for later reduction
    ol.vstore(processed, temp_ptr, offset, 4)
