use std::vec;
use rayon::prelude::*;

pub struct Matrix<mat_type> {
    pub data: Vec<Vec<mat_type>>,
    pub dim1: usize,
    pub dim2: usize,
    pub shape: Vec<usize>,
}


// Implement methods for the Matrix struct
impl<mat_type: Clone + Default + Copy> Matrix<mat_type> {
    // Constructors
    pub fn new(data: Vec<Vec<mat_type>>) -> Self {
        let dim1 = data.len();
        let dim2 = data[0].len();

        // Create a new matrix and return it
        Matrix { data, dim1, dim2, shape: vec![dim1, dim2] }
    }
    pub fn zeros(dim1: usize, dim2: usize) -> Self {
        let mut data: Vec<Vec<mat_type>> = vec![vec![mat_type::default(); dim2]; dim1];
        Matrix { data, dim1, dim2, shape: vec![dim1, dim2] }
    }

    // Getters and settings for each value
    pub fn get(&self, i: usize, j: usize) -> mat_type {
        self.data[i][j].clone()
    }
    pub fn set(&mut self, i: usize, j: usize, val: mat_type) {
        self.data[i][j] = val;
    }


    // Get the transpose of the matrix
    pub fn transpose(&self) -> Matrix<mat_type> {
        let mut result = Matrix::zeros(self.dim2, self.dim1);
        for i in 0..self.dim1 {
            for j in 0..self.dim2 {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }



    // Position-wise multiplication (Hadamard product)
    pub fn hadamard(&self, other: Matrix<mat_type>) -> Matrix<mat_type> 
    where
        mat_type: std::ops::Mul<Output = mat_type> + Clone,
    {
        // The matrices must have the same shape
        assert_eq!(self.shape, other.shape);

        // Create a new matrix of zeros
        let mut result = Matrix::zeros(self.dim1, self.dim2);

        // Perform the matrix addition
        for i in 0..self.dim1 {
            for j in 0..self.dim2 {
                result.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }

        // Return the result
        result
    }


    // Multiply the matrix, but transposed
    pub fn mult_transpose(&self, other: &Matrix<mat_type>) -> Matrix<mat_type> 
    where
        mat_type: std::ops::Mul<Output = mat_type> + std::ops::AddAssign + Default + Copy + Send + Sync,
    {
        // dim2 of the first matrix must be equal to dim2 of the second matrix
        assert_eq!(self.dim2, other.dim2);
        
        // Create a new matrix of zeros
        let mut result = Matrix::zeros(self.dim1, other.dim1);

        // Perform the matrix multiplication
        result.data.par_iter_mut().enumerate().for_each(|(i, row)| {
            for k in 0..other.dim2 {
                let temp = self.data[i][k];
                for j in 0..other.dim1 {
                    row[j] += temp * other.data[j][k];
                }
            }
        });

        // Return the result
        result
    }
}


// Implement the Display trait for the Matrix struct
impl<mat_type: std::fmt::Display> std::fmt::Display for Matrix<mat_type> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for i in 0..self.dim1 {
            for j in 0..self.dim2 {
                write!(f, "{} ", self.data[i][j])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}


// Overload the * operator for matrix multiplication
impl<mat_type> std::ops::Mul for &Matrix<mat_type> 
where
    mat_type: std::ops::Mul<Output = mat_type> + std::ops::AddAssign + Default + Copy + Send + Sync,
{
    // Define the output type
    type Output = Matrix<mat_type>;

    fn mul(self, other: &Matrix<mat_type>) -> Matrix<mat_type> {
        // dim2 of the first matrix must be equal to dim1 of the second matrix
        assert_eq!(self.dim2, other.dim1);
        
        // Create a new matrix of zeros
        let mut result = Matrix::zeros(self.dim1, other.dim2);

        // Perform the matrix multiplication
        result.data.par_iter_mut().enumerate().for_each(|(i, row)| {
            for k in 0..other.dim1 {
                let temp = self.data[i][k];
                for j in 0..other.dim2 {
                    row[j] += temp * other.data[k][j];
                }
            }
        });

        // Return the result
        result
    }
}


// Multiplication with a reference
impl<mat_type> std::ops::Mul<&Matrix<mat_type>> for Matrix<mat_type> 
where
    mat_type: std::ops::Mul<Output = mat_type> + std::ops::AddAssign + Default + Copy + Send + Sync,
{
    // Define the output type
    type Output = Matrix<mat_type>;

    fn mul(self, other: &Matrix<mat_type>) -> Matrix<mat_type> {
        // dim2 of the first matrix must be equal to dim1 of the second matrix
        assert_eq!(self.dim2, other.dim1);
        
        // Create a new matrix of zeros
        let mut result = Matrix::zeros(self.dim1, other.dim2);

        // Perform the matrix multiplication
        result.data.par_iter_mut().enumerate().for_each(|(i, row)| {
            for k in 0..other.dim1 {
                let temp = self.data[i][k];
                for j in 0..other.dim2 {
                    row[j] += temp * other.data[k][j];
                }
            }
        });

        // Return the result
        result
    }
}



// Overload the + operator for matrix addition
impl<mat_type> std::ops::Add for &Matrix<mat_type> 
where
    mat_type: std::ops::Add<Output = mat_type> + Default + Copy,
{
    // Define the output type
    type Output = Matrix<mat_type>;

    fn add(self, other: &Matrix<mat_type>) -> Matrix<mat_type> {
        // The matrices must have the same shape
        assert_eq!(self.shape[1], other.shape[1]);

        // Create a new matrix of zeros
        let mut result = Matrix::zeros(self.dim1, self.dim2);

        // Perform the matrix addition
        for i in 0..self.dim1 {
            for j in 0..self.dim2 {
                // Index is 0 if this is a vector
                if self.dim1 == 1 {
                    result.data[0][j] = self.data[0][j] + other.data[i][j];
                    continue;
                }
                else if other.dim1 == 1 {
                    result.data[i][j] = self.data[i][j] + other.data[0][j];
                    continue;
                }
                else {
                    result.data[i][j] = self.data[i][j] + other.data[i][j];
                }
            }
        }

        // Return the result
        result
    }
}




// Addition with a reference
impl<mat_type> std::ops::Add<&Matrix<mat_type>> for Matrix<mat_type> 
where
    mat_type: std::ops::Add<Output = mat_type> + Default + Copy,
{
    // Define the output type
    type Output = Matrix<mat_type>;

    fn add(self, other: &Matrix<mat_type>) -> Matrix<mat_type> {
        // The matrices must have the same shape
        assert_eq!(self.shape[1], other.shape[1]);

        // Create a new matrix of zeros
        let mut result = Matrix::zeros(self.dim1, self.dim2);

        // Perform the matrix addition
        for i in 0..self.dim1 {
            for j in 0..self.dim2 {
                // Index is 0 if this is a vector
                if self.dim1 == 1 {
                    result.data[0][j] = self.data[0][j] + other.data[i][j];
                    continue;
                }
                else if other.dim1 == 1 {
                    result.data[i][j] = self.data[i][j] + other.data[0][j];
                    continue;
                }
                else {
                    result.data[i][j] = self.data[i][j] + other.data[i][j];
                }
            }
        }

        // Return the result
        result
    }
}





// Overload the - operator for matrix subtraction
impl<mat_type> std::ops::Sub for &Matrix<mat_type> 
where
    mat_type: std::ops::Sub<Output = mat_type> + Default + Copy,
{
    // Define the output type
    type Output = Matrix<mat_type>;

    fn sub(self, other: &Matrix<mat_type>) -> Matrix<mat_type> {
        // The matrices must have the same shape
        assert_eq!(self.shape[1], other.shape[1]);

        // Create a new matrix of zeros
        let mut result = Matrix::zeros(self.dim1, self.dim2);

        // Perform the matrix addition
        for i in 0..self.dim1 {
            for j in 0..self.dim2 {
                // Index is 0 if this is a vector
                if self.dim1 == 1 {
                    result.data[0][j] = self.data[0][j] - other.data[i][j];
                    continue;
                }
                else if other.dim1 == 1 {
                    result.data[i][j] = self.data[i][j] - other.data[0][j];
                    continue;
                }
                else {
                    result.data[i][j] = self.data[i][j] - other.data[i][j];
                }
            }
        }

        // Return the result
        result
    }
}




// Subassign overload for the Matrix struct
impl<mat_type> std::ops::SubAssign for Matrix<mat_type> 
where
    mat_type: std::ops::SubAssign + Default + Copy,
{
    fn sub_assign(&mut self, other: Matrix<mat_type>) {
        // The matrices must have the same shape
        assert_eq!(self.shape[1], other.shape[1]);

        // Perform the matrix addition
        for i in 0..self.dim1 {
            for j in 0..self.dim2 {
                // Index is 0 if this is a vector
                if self.dim1 == 1 {
                    self.data[0][j] -= other.data[i][j];
                    continue;
                }
                else if other.dim1 == 1 {
                    self.data[i][j] -= other.data[0][j];
                    continue;
                }
                else {
                    self.data[i][j] -= other.data[i][j];
                }
            }
        }
    }
}



// Overload the [] operator for matrix indexing
impl<mat_type> std::ops::Index<usize> for Matrix<mat_type> {
    type Output = Vec<mat_type>;

    fn index(&self, i: usize) -> &Vec<mat_type> {
        &self.data[i]
    }
}

// Mutably index into the matrix
impl<mat_type> std::ops::IndexMut<usize> for Matrix<mat_type> {
    fn index_mut(&mut self, i: usize) -> &mut Vec<mat_type> {
        &mut self.data[i]
    }
}


// Copy overload for the Matrix struct
impl<mat_type: Clone> Clone for Matrix<mat_type> {
    fn clone(&self) -> Self {
        let mut data = Vec::new();
        for i in 0..self.dim1 {
            data.push(self.data[i].clone());
        }
        Matrix {
            data,
            dim1: self.dim1,
            dim2: self.dim2,
            shape: self.shape.clone(),
        }
    }
}