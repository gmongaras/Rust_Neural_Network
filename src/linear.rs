use rand::thread_rng;
use rand::distributions::{Uniform, Distribution};
use crate::matrix::Matrix;



pub struct Linear {
    pub weights: Matrix<f32>,
    pub bias: Matrix<f32>,
}


impl Linear {
    pub fn new(dim1: usize, dim2: usize) -> Self {
        let mut weights = Matrix::zeros(dim1, dim2);
        let mut bias = Matrix::zeros(1, dim2);

        // lower and uppwer bound
        let dim1_: f32 = dim1 as f32;
        let bound: f32 = 1.0 / dim1_.sqrt();
        // Define the range for the uniform distribution
        let range: Uniform<f32> = Uniform::new(-bound, bound);
        // Initialize a random number generator
        let mut rng: rand::prelude::ThreadRng = thread_rng();

        // Fill with random values
        for i in 0..dim1 {
            for j in 0..dim2 {
                // Fill with fp32 values
                weights[i][j] = range.sample(&mut rng) as f32;
            }
        }
        for i in 0..dim2 {
            bias[0][i] = range.sample(&mut rng) as f32;
        }

        Linear { weights, bias }
    }
    pub fn from_weights(weights: Matrix<f32>, bias: Matrix<f32>) -> Self {
        Linear { weights, bias }
    }

    // Forward is a matrix multiplication
    // Note that the weights are always transposed for the matmul
    pub fn forward(self, x: Matrix<f32>) -> Matrix<f32> {
        (x * self.weights) + self.bias
    }


    // Backward pass
    pub fn backward(self, x: Matrix<f32>, gradient: Matrix<f32>, learning_rate: f32, clip_value: f32) -> (Matrix<f32>, Linear) {
        // Compute the gradient for the weights
        let mut weights_gradient = x.transpose() * gradient.clone();
        let mut bias_gradient = Matrix::zeros(1, gradient.dim2);
        for i in 0..gradient.dim1 {
            for j in 0..gradient.dim2 {
                bias_gradient[0][j] += gradient[i][j];
            }
        }

        // Clip the gradients
        for i in 0..weights_gradient.dim1 {
            for j in 0..weights_gradient.dim2 {
                if weights_gradient[i][j] > clip_value {
                    weights_gradient[i][j] = clip_value;
                } else if weights_gradient[i][j] < -clip_value {
                    weights_gradient[i][j] = -clip_value;
                }
            }
        }
        for i in 0..bias_gradient.dim2 {
            if bias_gradient[0][i] > clip_value {
                bias_gradient[0][i] = clip_value;
            } else if bias_gradient[0][i] < -clip_value {
                bias_gradient[0][i] = -clip_value;
            }
        }

        // Multiple by the learning rate
        for i in 0..self.weights.dim1 {
            for j in 0..self.weights.dim2 {
                weights_gradient[i][j] *= learning_rate;
            }
        }
        for i in 0..self.bias.dim2 {
            bias_gradient[0][i] *= learning_rate;
        }

        // Compute the gradient for the next layer
        let next_gradient = gradient.clone() * self.weights.clone().transpose();

        // Update the weights and bias
        let mut new_linear = self.clone();
        new_linear.weights = self.weights.clone() - weights_gradient.clone();
        new_linear.bias = self.bias.clone() - bias_gradient.clone();

        (next_gradient, new_linear)
    }
}




// Overload the Copy trait for Linear
impl Clone for Linear {
    fn clone(&self) -> Self {
        Linear {
            weights: self.weights.clone(),
            bias: self.bias.clone(),
        }
    }
}