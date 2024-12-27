use crate::matrix::Matrix;
use std::cmp::max;








pub struct Nonlinearity {
    pub function_name: String,
}


impl Nonlinearity {
    // Constructor
    pub fn new(function_name: String) -> Self {
        Nonlinearity { function_name }
    }

    pub fn forward(self, x: Matrix<f32>) -> Matrix<f32>
    {
        match self.function_name.as_str() {
            "relu" => relu(x),
            "sigmoid" => sigmoid(x),
            _ => panic!("Unknown nonlinearity"),
        }
    }

    pub fn backward(self, x: Matrix<f32>) -> Matrix<f32>
    {
        match self.function_name.as_str() {
            "relu" => relu_backward(x),
            "sigmoid" => sigmoid_backward(x),
            _ => panic!("Unknown nonlinearity"),
        }
    }
}


// Overload the Copy trait for Nonlinearity
impl Clone for Nonlinearity {
    fn clone(&self) -> Self {
        Nonlinearity {
            function_name: self.function_name.clone(),
        }
    }
}




// ReLU nonlinearity applied per-element
fn relu(x: Matrix<f32>) -> Matrix<f32>
{
    let mut result: Matrix<f32> = Matrix::zeros(x.dim1, x.dim2);
    for i in 0..x.dim1 {
        for j in 0..x.dim2 {
            result[i][j] = if x[i][j] > 0.0 { x[i][j].clone() } else { 0.0 };
        }
    }
    result
}
// Sigmoid nonlinearity applied per-element
fn sigmoid(x: Matrix<f32>) -> Matrix<f32>
{
    let mut result: Matrix<f32> = Matrix::zeros(x.dim1, x.dim2);
    for i in 0..x.dim1 {
        for j in 0..x.dim2 {
            result[i][j] = 1.0 / (1.0 + (-x[i][j]).exp());
        }
    }
    result
}



// Backward ReLU nonlinearity applied per-element
fn relu_backward(x: Matrix<f32>) -> Matrix<f32>
{
    let mut result: Matrix<f32> = Matrix::zeros(x.dim1, x.dim2);
    for i in 0..x.dim1 {
        for j in 0..x.dim2 {
            result[i][j] = if x[i][j] > 0.0 { 1.0 } else { 0.0 };
        }
    }
    result
}
// Backward Sigmoid nonlinearity applied per-element
fn sigmoid_backward(x: Matrix<f32>) -> Matrix<f32>
{
    let mut result: Matrix<f32> = Matrix::zeros(x.dim1, x.dim2);
    for i in 0..x.dim1 {
        for j in 0..x.dim2 {
            result[i][j] = x[i][j] * (1.0 - x[i][j]);
        }
    }
    result
}