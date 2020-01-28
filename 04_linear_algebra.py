from typing import List

# The simplest from-scratch approach is to represent vectors as lists
# of numbers. We'll make a type alias that says a Vector is just a 
# list of floats.
Vector = List[float]

height_weight_age = [70,    # inches
                     170,   # pounds
                     40 ]   # years

grades = [95,   # exam 1
          80,   # exam 2
          75,   # exam 3
          62]   # exam 4

# We want to perform arithmetic on these vectors. But python lists aren't 
# vectors and have no built-in arithmetic facilities. We need to build these 
# ourselves.

# Also - lists are a great way to explain the concepts here but this isn't 
# going to perfrom well enough for production code. There, we'd want to use
# NumPy's high performance arrays instead, with their built in arithmetic
# operations that have been optimized for speed.

# Addition

# We'll frequently need to add two vectors. Vectors add componentwise.
# We can easily implement this by zip-ing the vectors together and using a 
# list comprehension to add the corresponding elements.

def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements."""
    assert len(v) == len(w), "vectors must be the same length"
    return [v_i + w_i for v_i, w_i in zip(v, w)]

assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

# Subtraction

# Similarly, we need to subtract two vectors - by subtracting the
# corresponding elements.

def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts corresponding elements."""
    assert len(v) == len(w), "vectors must be the same length"
    return [v_i - w_i for v_i, w_i in zip(v, w)]

assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]

# Addition: List of Vectors

# Sometimes we need to do a sum across a whole list of vectors. This is going 
# to be like add but for an arbitrary length list.

def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements."""
    # Check that vectors is not empty
    assert vectors, "no vectors provided"

    # Check that the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes"

    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors)
                for i in range(num_elements)]

assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]

# Scalar Multiplication

# We also need to be able to multiply a vector by a scalar.
# This is simply multiplying each item by that number.

def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies each element by c"""
    return [c * v_i for v_i in v]

assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]

# Now we can compute the componentwise means of a list of (same-sized) 
# vectors.
def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]

# Dot Product

# The dot product of two vectors is the sum of their componentwise products.
def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be the same length"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1, 2, 3], [4, 5, 6]) == 32

# Given the dot product function, it's easy to compute a vectors sum of 
# squares.
def sum_of_squares(v: Vector) -> float:
    """Computes v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

assert sum_of_squares([1, 2, 3]) == 14

# We can use the sum of squares to calculate length, called _magnitude_
import math

def magnitude(v: Vector) -> float:
    """Returns the magnitude (or length) of v"""
    return math.sqrt(sum_of_squares(v))

assert magnitude([3, 4]) == 5

# We now have all the pieces we need to compute the distance between two 
# vectors. The definition is sqrt((v_1 - w_1)^2 + ... + (v_n - w_n)^2)

def squared_distance(v: Vector, w: Vector) -> float:
    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
    """Computes the distance between v and w"""
    return magnitude(subtract(v, w))



# A matrix is a two-dimensional collection of numbers.
# We can think about it as a list of lists, with each inner list having the 
# same size and representing one row of the matrix.
# If A is a matrix, then A[i][j] is the element in the ith row and jth column.
# Python lists are zero-indexed, so the first row is called "row 0".

Matrix = List[List[float]]

A = [[1, 2, 3],         # A has 2 rows and 3 columns (each list is a row)
     [4, 5, 6]]

B = [[1, 2],            # B has 3 rows and 2 columns
     [3, 4],
     [5, 6]]

# Given this representation, the matrix A has len(A) rows and len(A[0]) 
# columns, called the "shape" of the matrix.

from typing import Tuple

def shape(A: Matrix) -> Tuple[int, int]:
    """Returns (# of rows of A, # of columns of A)"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols

assert shape(A) == (2, 3)
assert shape(B) == (3, 2)

# If a matrix has n rows and k columns, we will call it an n by k (n x k)
# matrix. We can think of each row of an n x k matrix as a vector of length
# k, and we can think of each column of an n x k matrix as a vector of length
# n. 

def get_row(A: Matrix, i: int) -> Vector:
    """Returns the i-th row of A as a Vector"""
    return A[i]

def get_column(A: Matrix, j: int) -> Vector:
    """Returns the j-th column of A as a Vector"""
    return [A_i[j] for A_i in A]


# We want to be able to create a matrix. We'll write a function that takes a
# shape (number of rows and columns) and a function for generalizing it's
# elements. Here, we can take advantage of functions as first class objects
# in python.
# Our generating function will take a row number and column number as ints
# and return a float for the value at that row and column.

from typing import Callable

def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    """
    Returns a num_rows x num_cols matrix
    whose (i,j)-th entry is entry_fn(i, j)
    """
    return [[entry_fn(i, j)
             for j in range(num_cols)]
            for i in range(num_rows)]

# As an example, you could use this function to make a 5x5 identity matrix.

def identity_matrix(n: int) -> Matrix:
    """Returns the n x n identity matrix"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]]

