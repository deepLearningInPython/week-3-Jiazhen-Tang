import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.


# Task 1: Compute Output Size for 1D Convolution
# Instructions:
# Write a function that takes two one-dimensional numpy arrays (input_array, kernel_array) as arguments.
# The function should return the length of the convolution output (assuming no padding and a stride of one).
# The output length can be computed as follows:
# (input_length - kernel_length + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_1d(input_array, kernel_array):
    # Get the lengths of the input and kernel arrays
    input_length = len(input_array)
    kernel_length = len(kernel_array)
    
    # Compute the output length using the formula: (input_length - kernel_length + 1)
    output_length = input_length - kernel_length + 1
    
    return output_length


# -----------------------------------------------
# Example:
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(compute_output_size_1d(input_array, kernel_array))


# Task 2: 1D Convolution
# Instructions:
# Write a function that takes a one-dimensional numpy array (input_array) and a one-dimensional kernel array (kernel_array)
# and returns their convolution (no padding, stride 1).

# Your code here:
# -----------------------------------------------

def convolve_1d(input_array, kernel_array):
    # Calculate the expected output length
    input_length = len(input_array)
    kernel_length = len(kernel_array)
    output_length = input_length - kernel_length + 1
    
    # Initialize the output array with zeros
    output_array = np.zeros(output_length)
    
    # Loop through the input array to compute the convolution
    for i in range(output_length):
        # Slice the input array to match the kernel size
        # The slice goes from index i to i + kernel_length
        input_segment = input_array[i : i + kernel_length]
        
        # Perform element-wise multiplication and sum the result (dot product)
        output_array[i] = np.sum(input_segment * kernel_array)
        
    return output_array

# -----------------------------------------------
# Another tip: write test cases like this, so you can easily test your function.
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(convolve_1d(input_array, kernel_array))

# Task 3: Compute Output Size for 2D Convolution
# Instructions:
# Write a function that takes two two-dimensional numpy matrices (input_matrix, kernel_matrix) as arguments.
# The function should return a tuple with the dimensions of the convolution of both matrices.
# The dimensions of the output (assuming no padding and a stride of one) can be computed as follows:
# (input_height - kernel_height + 1, input_width - kernel_width + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_2d(input_matrix, kernel_matrix):
    # Get the dimensions (rows, cols) of input and kernel
    input_rows, input_cols = input_matrix.shape
    kernel_rows, kernel_cols = kernel_matrix.shape
    
    # Calculate output dimensions using the formula: Input - Kernel + 1
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1
    
    return (output_rows, output_cols)


# -----------------------------------------------


# Task 4: 2D Convolution
# Instructions:
# Write a function that computes the convolution (no padding, stride 1) of two matrices (input_matrix, kernel_matrix).
# Your function will likely use lots of looping and you can reuse the functions you made above.

# Your code here:
# -----------------------------------------------
def convolute_2d(input_matrix, kernel_matrix):
    # Get dimensions of input and kernel
    input_rows, input_cols = input_matrix.shape
    kernel_rows, kernel_cols = kernel_matrix.shape
    
    # Calculate the dimensions of the output matrix
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1
    
    # Initialize the output matrix with zeros
    output_matrix = np.zeros((output_rows, output_cols))
    
    # Loop over the rows of the output
    for i in range(output_rows):
        # Loop over the columns of the output
        for j in range(output_cols):
            # Define the slice of the input matrix
            # Rows from i to i + kernel_rows
            # Cols from j to j + kernel_cols
            input_patch = input_matrix[i : i + kernel_rows, j : j + kernel_cols]
            
            # Perform element-wise multiplication and sum (dot product)
            output_matrix[i, j] = np.sum(input_patch * kernel_matrix)
            
    return output_matrix


# -----------------------------------------------