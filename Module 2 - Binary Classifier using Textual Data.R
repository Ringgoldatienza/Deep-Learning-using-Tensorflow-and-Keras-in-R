library(keras)

#Load dataset
imdb <- dataset_imdb(num_words = 10000)

#Separate training and test set
train_data <- imdb$train$x
train_labels <- imdb$train$y
test_data <- imdb$test$x
test_labels <- imdb$test$y

#We can decode reviews back to text using these lines of code
word_index <- dataset_imdb_word_index()

reverse_word_index <- names(word_index)
names(reverse_word_index) <- as.character(word_index)

decoded_words <- train_data[[1]] %>%
  sapply(function(i) {
    if (i > 3) reverse_word_index[[as.character(i - 3)]]
    else "?"
  })

decoded_review <- paste0(decoded_words, collapse = " ")
cat(decoded_review, "\n")

#Encode the integer sequences via multi-hot encoding
vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- array(0, dim = c(length(sequences), dimension))
  for (i in seq_along(sequences)) {
    sequence <- sequences[[i]]
    for (j in sequence)
      results[i, j] <- 1
  }
  results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

#We also need to vectorize labers
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

#Build the model
model <- keras_model_sequential() %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(1, activation = "sigmoid")

#Compile the model
#We set "rmsprop" as optimizer as this is the usual good default choice for virtually any problem
#We set binary_crossentropy because it is the best choice when dealing with models that output probabilities
#We also set accuracy
model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")

#Validate the approach by setting asside a validation set
x_val <- x_train[seq(10000), ]
partial_x_train <- x_train[-seq(10000), ]
y_val <- y_train[seq(10000)]
partial_y_train <- y_train[-seq(10000)]

#Train the model
#We set 20 epochs (20 iterations over all samples in the training data) in minibatches of 512 samples
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

#The results shows overfitting (the model shows good performance on training but not on validation)

#Retrain model from scratch
#Build the model
model <- keras_model_sequential() %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(1, activation = "sigmoid")

model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")

model %>% fit(x_train, y_train, epochs = 4, batch_size = 512)
results <- model %>% evaluate(x_test, y_test) #We get loss = .29 and accuracy = .88.

#Generate prediction
predictions <- model %>% predict(x_test)
