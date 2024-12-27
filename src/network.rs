use crate::linear::Linear;
use crate::matrix::Matrix;
use crate::nonlinearity::Nonlinearity;





pub struct Network {
    pub input_size: usize,
    pub output_size: usize,
    pub hidden_sizes: Vec<usize>,
    pub nonlinearity: Nonlinearity,
    pub layers: Vec<Linear>,
    pub intermediates_prelayer: Vec<Matrix<f32>>,
    pub intermediates_preactivation: Vec<Matrix<f32>>,
}



// Defaults
impl Default for Network {
    fn default() -> Self {
        Network {
            input_size: 0,
            output_size: 0,
            hidden_sizes: Vec::new(),
            nonlinearity: Nonlinearity::new("relu".to_string()),
            layers: Vec::new(),
            intermediates_prelayer: Vec::new(),
            intermediates_preactivation: Vec::new(),
        }
    }
}



impl Network {
    // Constructor
    pub fn new(input_size: usize, output_size: usize, hidden_sizes: Vec<usize>, nonlinearity: Nonlinearity) -> Self {
        // num_layers must be at least 2
        assert!(input_size > 0);
        assert!(output_size > 0);
        assert!(hidden_sizes.len() > 0);

        // Create the network
        let mut layers: Vec<Linear> = Vec::new();
        let mut prev_size: usize = input_size;
        for i in 0..hidden_sizes.len() {
            let layer: Linear = Linear::new(prev_size, hidden_sizes[i]);
            layers.push(layer);
            prev_size = hidden_sizes[i];
        }

        // Add the output layer
        let output_layer: Linear = Linear::new(prev_size, output_size);
        layers.push(output_layer);

        // Return the network
        Network { input_size, output_size, hidden_sizes, nonlinearity, layers, intermediates_prelayer: Vec::new(), intermediates_preactivation: Vec::new() }
    }

    // Forward pass
    pub fn forward(&mut self, x: Matrix<f32>) -> Matrix<f32> {
        // Perform the forward pass
        let mut result: Matrix<f32> = x.clone();
        let mut intermediates_prelayer: Vec<Matrix<f32>> = Vec::new(); // Saves all intermediate results
        for i in 0..self.layers.len() {
            // Save the intermediate result (before the linear layer)
            intermediates_prelayer.push(result.clone());

            result = self.layers[i].clone().forward(result);

            // Save the intermediate result (before the nonlinearity)
            self.intermediates_preactivation.push(result.clone());

            // Apply the nonlinearity
            if i < self.layers.len() - 1 { // Only apply nonlinearity to hidden layers
                result = self.nonlinearity.clone().forward(result);
            }
        }

        // Save the intermediates
        self.intermediates_prelayer = intermediates_prelayer.clone();
        self.intermediates_preactivation = self.intermediates_preactivation.clone();

        result
    }


    // Backward pass
    pub fn backward(&mut self, output: Matrix<f32>, y: Matrix<f32>, learning_rate: f32, clip_value: f32) {
        // Perform the backward pass
        let mut gradient: Matrix<f32> = self.cross_entropy_loss_gradient(output, y.clone());
        for i in (0..self.layers.len()).rev() {

            // If not the last layer, then we need apply the backward nonlinearity
            if i < self.layers.len() - 1 {
                let act_grad = self.nonlinearity.clone().backward(self.intermediates_preactivation[i].clone());

                // Perform a Hadamard product with the gradient
                gradient = gradient.hadamard(act_grad.clone());
            }

            // Get the intermediates at this layer
            let intermediates = self.intermediates_prelayer[i].clone();

            // Update the weights and get the gradient wrt. X
            let (next_gradient, new_layer) = self.layers[i].clone().backward(intermediates, gradient.clone(), learning_rate, clip_value);
            self.layers[i] = new_layer;

            // Update the gradient for the next layer
            gradient = next_gradient.clone();
        }
    }


    // Cross entropy loss function
    pub fn cross_entropy_loss(&self, output: Matrix<f32>, y: Matrix<f32>) -> f32 {
        // Compute the cross entropy loss
        let mut loss: f32 = 0.0;
        for i in 0..output.dim1 { // Iterate over the batch
            // We need to get the denominator of the softmax
            let mut sum: f32 = 0.0;
            for j in 0..output.dim2 {
                sum += output[i][j].exp();
            }
            // Get the class probability for the correct class
            let prob = output[i][y[i][0] as usize].exp() / sum;

            // We only get the loss for the correct class
            loss += - 1.0 * prob.ln();
        }
        // Return the average loss
        loss / output.dim1 as f32
    }



    pub fn cross_entropy_loss_gradient(&self, output: Matrix<f32>, y: Matrix<f32>) -> Matrix<f32> {
        // Initialize a matrix to store the gradients
        let mut gradient = Matrix::zeros(output.dim1, output.dim2);
    
        for i in 0..output.dim1 { // Iterate over the batch
            // Find the maximum value in the row for numerical stability
            let max_logit = output[i].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
            // Compute the softmax denominator using the stable trick
            let mut sum: f32 = 0.0;
            for j in 0..output.dim2 {
                sum += (output[i][j] - max_logit).exp();
            }
    
            // Compute the gradient for each class
            for j in 0..output.dim2 {
                let softmax = (output[i][j] - max_logit).exp() / sum;
    
                if j == y[i][0] as usize {
                    // Subtract 1 from the softmax for the correct class
                    gradient[i][j] = softmax - 1.0;
                } else {
                    gradient[i][j] = softmax;
                }
            }
        }
    
        gradient
    }
    
}