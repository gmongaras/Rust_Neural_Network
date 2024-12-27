use std::io;
use std::io::prelude::*;

// Declare modules
mod matrix;
mod linear;
mod nonlinearity;
mod network;

// Import modules
use matrix::Matrix;
use linear::Linear;
use network::Network;

use rand::seq::SliceRandom;
use rand::thread_rng;

fn shuffle_data_and_labels(train_data: &mut Vec<Vec<f32>>, train_labels: &mut Vec<Vec<f32>>) {
    assert_eq!(train_data.len(), train_labels.len(), "Data and labels must have the same length!");

    // Create a vector of indices and shuffle it
    let mut rng = thread_rng();
    let mut indices: Vec<usize> = (0..train_data.len()).collect();
    indices.shuffle(&mut rng);

    // Reorder both train_data and train_labels based on shuffled indices
    let mut shuffled_data = vec![Vec::new(); train_data.len()];
    let mut shuffled_labels = vec![Vec::new(); train_labels.len()];
    for (new_index, &old_index) in indices.iter().enumerate() {
        shuffled_data[new_index] = train_data[old_index].clone();
        shuffled_labels[new_index] = train_labels[old_index].clone();
    }

    // Replace original data and labels with shuffled versions
    *train_data = shuffled_data;
    *train_labels = shuffled_labels;
}



fn calculate_accuracy(output: Matrix<f32>, labels: Matrix<f32>) -> f32 {
    let mut correct: f32 = 0.0;
    let mut total: f32 = 0.0;

    // Iterate over the entire batch
    for i in 0..output.dim1 {

        // Get the max and argmax for this part of the batch
        let mut max_index: usize = 0;
        let mut max_value: f32 = 0.0;
        for j in 0..output.dim2 {
            if output[i][j] > max_value {
                max_value = output[i][j];
                max_index = j;
            }
        }

        // Increase total and correct if the argmax is correct
        if (labels[i][0] as usize) == max_index {
            correct += 1.0;
        }
        total += 1.0;
    }

    // Return the accuracy
    correct / total
}



fn main() {
    // Load in thw MNIST data
    let mut rdr_train = csv::Reader::from_path("mnist_train.csv").unwrap();
    let mut rdr_test = csv::Reader::from_path("mnist_test.csv").unwrap();
    
    // Read the training data
    let mut train_data: Vec<Vec<f32>> = Vec::new();
    let mut train_labels: Vec<Vec<f32>> = Vec::new();
    for result in rdr_train.records() {
        let record = result.unwrap();
        let mut data: Vec<f32> = Vec::new();
        let mut labels: Vec<f32> = Vec::new();
        for (i, value) in record.iter().enumerate() {
            if i == 0 {
                labels.push(value.parse::<f32>().unwrap());
            }
            else {
                data.push(value.parse::<f32>().unwrap() / 255.0);
            }
        }
        train_data.push(data);
        train_labels.push(labels);
    }

    // Read the test data
    let mut test_data: Vec<Vec<f32>> = Vec::new();
    let mut test_labels: Vec<Vec<f32>> = Vec::new();
    for result in rdr_test.records() {
        let record = result.unwrap();
        let mut data: Vec<f32> = Vec::new();
        let mut labels: Vec<f32> = Vec::new();
        for (i, value) in record.iter().enumerate() {
            if i == 0 {
                labels.push(value.parse::<f32>().unwrap());
            }
            else {
                data.push(value.parse::<f32>().unwrap() / 255.0);
            }
        }
        test_data.push(data);
        test_labels.push(labels);
    }

    // Print out the first 10 labels
    for i in 0..10 {
        println!("{:?}", train_labels[i]);
        println!("{:?}", train_data[i]);
    }

    // Size of the train data and test data
    let train_data_size = train_data.len();
    let test_data_size = test_data.len();
    let image_size = train_data[0].len();
    println!("Train data size: {}", train_data_size);
    println!("Test data size: {}", test_data_size);
    println!("Image size: {}", image_size);


    // All parameters
    let network_layers = vec![512, 128];
    let num_classes = 10;
    let nonlinearity_str = "relu";
    let batch_size = 512;
    let learning_rate = 0.0005;
    let num_epochs = 10;
    let clip_value = 5.0;


    // Network
    let mut nonlinearity: nonlinearity::Nonlinearity = nonlinearity::Nonlinearity::new(nonlinearity_str.to_string());
    let mut model = Network::new(image_size, num_classes, network_layers, nonlinearity);

    // Training loop
    let num_batches: usize = train_data_size / batch_size;
    let num_batches_test: usize = test_data_size / batch_size;
    for epoch in 0..num_epochs {
        // Shuffle the data
        shuffle_data_and_labels(&mut train_data, &mut train_labels);

        // Total loss and accuracy
        let mut total_loss: f32 = 0.0;
        let mut total_accuracy: f32 = 0.0;

        for batch in 0..num_batches {
            // Get the batch data
            let mut batch_data: Vec<Vec<f32>> = Vec::new();
            let mut batch_labels: Vec<Vec<f32>> = Vec::new();
            for i in 0..batch_size {
                let index = batch * batch_size + i;
                batch_data.push(train_data[index].clone());
                batch_labels.push(train_labels[index].clone());
            }

            // Convert to matrices
            let x: Matrix<f32> = Matrix::new(batch_data.clone());
            let y: Matrix<f32> = Matrix::new(batch_labels.clone());

            // Forward pass
            let mut output: Matrix<f32> = model.forward(x.clone());

            // Get the loss
            let loss: f32 = model.cross_entropy_loss(output.clone(), y.clone());

            // Backward pass
            model.backward(output.clone(), y.clone(), learning_rate, clip_value);

            // Accuracy
            let accuracy: f32 = calculate_accuracy(output.clone(), y.clone());

            // Update the total loss and accuracy
            total_loss += loss;
            total_accuracy += accuracy;

            println!("Epoch: {}, Batch: {}, Loss: {}, Accuracy: {}", epoch, batch, loss, accuracy);
        }

        // Total loss and accuracy
        println!("Epoch {}, Total Loss: {}, Total Accuracy: {}", epoch, total_loss/num_batches as f32, total_accuracy/num_batches as f32);




        // Iterate over the test data and calculate the accuracy and loss
        let mut total_loss: f32 = 0.0;
        let mut total_accuracy: f32 = 0.0;
        for batch in 0..num_batches_test {
            // Get the batch data
            let mut batch_data: Vec<Vec<f32>> = Vec::new();
            let mut batch_labels: Vec<Vec<f32>> = Vec::new();
            for i in 0..batch_size {
                let index = batch * batch_size + i;
                batch_data.push(test_data[index].clone());
                batch_labels.push(test_labels[index].clone());
            }

            // Convert to matrices
            let x: Matrix<f32> = Matrix::new(batch_data.clone());
            let y: Matrix<f32> = Matrix::new(batch_labels.clone());

            // Forward pass
            let output: Matrix<f32> = model.forward(x.clone());

            // Get the loss
            let loss: f32 = model.cross_entropy_loss(output.clone(), y.clone());

            // Accuracy
            let accuracy: f32 = calculate_accuracy(output.clone(), y.clone());

            total_loss += loss;
            total_accuracy += accuracy;
        }

        // Total loss and accuracy
        println!("Epoch {}, Test Loss: {}, Test Accuracy: {}", epoch, total_loss/num_batches_test as f32, total_accuracy/num_batches_test as f32);

        println!("");
        println!("");        

    }



}