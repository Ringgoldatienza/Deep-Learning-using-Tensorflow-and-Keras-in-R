#Load libraries
library(tensorflow)
library(keras)
library(reticulate)

#Load dataset
reuters <- dataset_reuters(num_words = 10000)

#Separate train and test set
#Separate training and test set
train_data <- reuters$train$x
train_labels <- reuters$train$y
test_data <- reuters$test$x
test_labels <- reuters$test$y

#Decode newswires back to thext
word_index <- dataset_reuters_word_index()

reverse_word_index <- names(word_index)
names(reverse_word_index) <- as.character(word_index)

decoded_words <- train_data[[1]] %>%
  sapply(function(i) {
    if (i > 3) reverse_word_index[[as.character(i - 3)]]
    else "?"
  })
decoded_review <- paste0(decoded_words, collapse = " ")
decoded_review

#Encode the input data
#We use one-hot encoding(also called categorical encoding). It is widely used format for categorical data.
vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in seq_along(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

#Encode the labels
to_one_hot <- function(labels, dimension = 46) {
  results <- matrix(0, nrow = length(labels), ncol = dimension)
  labels <- labels + 1
  for(i in seq_along(labels)) {
    j <- labels[[i]]
    results[i, j] <- 1
  }
  results
}
y_train <- to_one_hot(train_labels)
y_test <- to_one_hot(test_labels)

#Built-in ways to do this in Keras use: 
#y_train <- to_categorical(train_labels)
#y_test <- to_categorical(test_labels)

#Build the model
#We use 64 classes for 1st and 2nd layers (higher that 46 output classes)
model <- keras_model_sequential() %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(46, activation = "softmax")

#Compile the model
model %>% compile(optimizer = "rmsprop",
                  loss = "categorical_crossentropy",
                  metrics = "accuracy")

#Validate the approach by setting asside a validation set
val_indices <- 1:1000

x_val <- x_train[val_indices, ]
partial_x_train <- x_train[-val_indices, ]

y_val <- y_train[val_indices, ]
partial_y_train <- y_train[-val_indices, ]

#Train the model
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

which.max(history$metrics$val_accuracy)

#Retrain model from scratch
model <- keras_model_sequential() %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(46, activation = "softmax")

model %>% compile(optimizer = "rmsprop",
                  loss = "categorical_crossentropy",
                  metrics = "accuracy")

model %>% fit(x_train, y_train, epochs = 6, batch_size = 512)

results <- model %>% evaluate(x_test, y_test) #loss: 1.1998 - accuracy: 0.7903

#Test how a random classifier would score
mean(test_labels == sample(test_labels)) #accuracy: 0.1932324

#Generate prediction
predictions <- model %>% predict(x_test)
