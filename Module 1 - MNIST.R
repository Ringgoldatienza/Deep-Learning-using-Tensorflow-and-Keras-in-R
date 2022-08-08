#Deep Learning with R

#Install tesnsorflow and keras packages
install.packages("tensorflow")
library(tensorflow)
install_tensorflow(envname = "r-reticulate")

#Install keras
keras::install_keras()
library(keras)

#Loading the MNIST dataset in Keras
mnist <- dataset_mnist()

train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

#Build the network architecture
model <- keras_model_sequential(list(
  layer_dense(units = 512, activation = "relu"),
  layer_dense(units = 10, activation = "softmax")
))

#Compilation step
compile(model,
        optimizer = "rmsprop",
        loss = "sparse_categorical_crossentropy",
        metrics = "accuracy")

#Prepare the image data
train_images <- array_reshape(train_images, c(60000, 28 * 28))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28 * 28))
test_images <- test_images / 255

#Fitting the model
fit(model, train_images, train_labels, epochs = 5, batch_size = 128)

#Use the model to make predictions
test_digits <- test_images[1:10, ]
predictions <- predict(model, test_digits)
str(predictions)

predictions[1, ]

which.max(predictions[1, ])

predictions[1, 8]

test_labels[1]

#Evaluate the model on new data
metrics <- evaluate(model, test_images, test_labels)
metrics["accuracy"]

################################################################################
#Reimplementing from scratch in TensorFlow

#A Simple Dense Class
layer_naive_dense <- function(input_size, output_size, activation) {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "NaiveDense"
  
  self$activation <- activation
  
  w_shape <- c(input_size, output_size)
  w_initial_value <- random_array(w_shape, min = 0, max = 1e-1)
  self$W <- tf$Variable(w_initial_value)
  
  b_shape <- c(output_size)
  
  b_initial_value <- array(0, b_shape)
  self$b <- tf$Variable(b_initial_value)
  
  self$weights <- list(self$W, self$b)
  
  self$call <- function(inputs) {
    self$activation(tf$matmul(inputs, self$W) + self$b)
  }
  self
}
  
#A Simple Sequential Class
naive_model_sequential <- function(layers) {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "NaiveSequential"
  
  self$layers <- layers
  
  weights <- lapply(layers, function(layer) layer$weights)
  
  self$weights <- do.call(c, weights)
  
  self$call <- function(inputs) {
    x <- inputs
    for (layer in self$layers)
      x <- layer$call(x)
    x
  }
  self
}

#Using this NaiveDense class and this NaiveSequential class, we can create a mock Keras model:
model <- naive_model_sequential(list(
  layer_naive_dense(input_size = 28 * 28, output_size = 512,
                    activation = tf$nn$relu),
  layer_naive_dense(input_size = 512, output_size = 10,
                    activation = tf$nn$softmax)
))

stopifnot(length(model$weights) == 4)

#Batch Generator
new_batch_generator <- function(images, labels, batch_size = 128) {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "BatchGenerator"
  
  stopifnot(nrow(images) == nrow(labels))
  self$index <- 1
  self$images <- images
  self$labels <- labels
  self$batch_size <- batch_size
  self$num_batches <- ceiling(nrow(images) / batch_size)
  self$get_next_batch <- function() {
    start <- self$index
    if(start > nrow(images))
      return(NULL)

    end <- start + self$batch_size - 1
    if(end > nrow(images))
      end <- nrow(images)
    
    self$index <- end + 1
    indices <- start:end
    list(images = self$images[indices, ],
         labels = self$labels[indices])
  }
  self
}

#Run one training step
one_training_step <- function(model, images_batch, labels_batch) {
  with(tf$GradientTape() %as% tape, {
    predictions <- model$call(images_batch)
    per_sample_losses <-
      loss_sparse_categorical_crossentropy(labels_batch, predictions)
    average_loss <- mean(per_sample_losses)
  })
  
  gradients <- tape$gradient(average_loss, model$weights)
  update_weights(gradients, model$weights)
  average_loss
}

learning_rate <- 1e-3

update_weights <- function(gradients, weights) {
  stopifnot(length(gradients) == length(weights))
  for (i in seq_along(weights))
    weights[[i]]$assign_sub(
      gradients[[i]] * learning_rate)
}

optimizer <- optimizer_sgd(learning_rate = 1e-3)

update_weights <- function(gradients, weights)
  optimizer$apply_gradients(zip_lists(gradients, weights))

str(zip_lists(
  gradients = list("grad_for_wt_1", "grad_for_wt_2", "grad_for_wt_3"),
  weights = list("weight_1", "weight_2", "weight_3")))

#Fit full training loop
fit <- function(model, images, labels, epochs, batch_size = 128) {
  for (epoch_counter in seq_len(epochs)) {
    cat("Epoch ", epoch_counter, "\n")
    batch_generator <- new_batch_generator(images, labels)
    for (batch_counter in seq_len(batch_generator$num_batches)) {
      batch <- batch_generator$get_next_batch()
      loss <- one_training_step(model, batch$images, batch$labels)
      if (batch_counter %% 100 == 0)
        cat(sprintf("loss at batch %s: %.2f\n", batch_counter, loss))
    }
  }
}

#Test model
mnist <- dataset_mnist()
train_images <- array_reshape(mnist$train$x, c(60000, 28 * 28)) / 255
test_images <- array_reshape(mnist$test$x, c(10000, 28 * 28)) / 255
test_labels <- mnist$test$y
train_labels <- mnist$train$y

fit(model, train_images, train_labels, epochs = 10, batch_size = 128)

#Evaluate model
predictions <- model$call(test_images)
predictions <- as.array(predictions)
predicted_labels <- max.col(predictions) - 1

matches <- predicted_labels == test_labels
cat(sprintf("accuracy: %.2f\n", mean(matches)))