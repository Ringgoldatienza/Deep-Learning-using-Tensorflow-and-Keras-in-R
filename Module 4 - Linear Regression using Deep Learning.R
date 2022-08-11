#Load libraries
library(tensorflow)
library(keras)
library(reticulate)

#Load dataset
boston <- dataset_boston_housing()

#Separate train and test sets
train_data <- boston$train$x
train_targets <- boston$train$y
test_data <- boston$test$x
test_targets <- boston$test$y

#Convert data into numeric values
x_train <- x_train %>%
  mutate(destination = as.factor(destination)) %>%
  mutate(passanger = as.factor(passanger)) %>%
  mutate(weather =  as.factor(weather)) %>%
  mutate(time = as.factor(time)) %>%
  mutate(coupon = as.factor(coupon)) %>%
  mutate(expiration = as.factor(expiration)) %>%
  mutate(gender = as.factor(gender)) %>%
  mutate(age = as.factor(age)) %>%
  mutate(maritalStatus = as.factor(maritalStatus)) %>%
  mutate(education = as.factor(education)) %>%
  mutate(occupation = as.factor(occupation)) %>%
  mutate(income = as.factor(income)) %>%
  mutate(Bar = as.factor(Bar)) %>%
  mutate(CoffeeHouse = as.factor(CoffeeHouse)) %>%
  mutate(CarryAway = as.factor(CarryAway)) %>%
  mutate(RestaurantLessThan20 = as.factor(RestaurantLessThan20)) %>%
  mutate(Restaurant20To50 = as.factor(Restaurant20To50))
x_train <- sapply(x_train, unclass)

x_test <- x_test %>%
  mutate(destination = as.factor(destination)) %>%
  mutate(passanger = as.factor(passanger)) %>%
  mutate(weather =  as.factor(weather)) %>%
  mutate(time = as.factor(time)) %>%
  mutate(coupon = as.factor(coupon)) %>%
  mutate(expiration = as.factor(expiration)) %>%
  mutate(gender = as.factor(gender)) %>%
  mutate(age = as.factor(age)) %>%
  mutate(maritalStatus = as.factor(maritalStatus)) %>%
  mutate(education = as.factor(education)) %>%
  mutate(occupation = as.factor(occupation)) %>%
  mutate(income = as.factor(income)) %>%
  mutate(Bar = as.factor(Bar)) %>%
  mutate(CoffeeHouse = as.factor(CoffeeHouse)) %>%
  mutate(CarryAway = as.factor(CarryAway)) %>%
  mutate(RestaurantLessThan20 = as.factor(RestaurantLessThan20)) %>%
  mutate(Restaurant20To50 = as.factor(Restaurant20To50))
x_test <- sapply(x_test, unclass)

#Normalize the data
mean <- apply(train_data, 2, mean)
sd <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = sd)
test_data <- scale(test_data, center = mean, scale = sd)

#Model definition
build_model <- function() {
  model <- keras_model_sequential() %>%
    layer_dense(64, activation = "relu") %>%
    layer_dense(64, activation = "relu") %>%
    layer_dense(1) #the model ends with a single unit and no activation (a typical setup for scalar regression)
  
  model %>% compile(optimizer = "rmsprop",
                    loss = "mse", #we use mse as this is the widely used loss function for regression problems.
                    metrics = "mae")
  model
}

#Validate approach using K-fold validation
k <- 4
fold_id <- sample(rep(1:k, length.out = nrow(train_data)))
num_epochs <- 100
all_scores <- numeric()

for (i in 1:k) {
  cat("Processing fold #", i, "\n")
  val_indices <- which(fold_id == i)
  
  val_data <- train_data[val_indices, ]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices, ]
  partial_train_targets <- train_targets[-val_indices]
  
  model <- build_model()
  
  model %>% fit(
    partial_train_data,
    partial_train_targets,
    epochs = num_epochs,
    batch_size = 16,
    verbose = 0
  )
  results <- model %>%
    evaluate(val_data, val_targets, verbose = 0)
  all_scores[[i]] <- results[['mae']]
}

all_scores #2.551575 2.228184 2.551098 2.205640

mean(all_scores) #2.384124

#Train model with 500 epochs
#Save the validation logs at each fold
num_epochs <- 500
all_mae_histories <- list()
for (i in 1:k) {
  cat("Processing fold #", i, "\n")
  val_indices <- which(fold_id == i)
  val_data <- train_data[val_indices, ]
  val_targets <- train_targets[val_indices]
  partial_train_data <- train_data[-val_indices, ] 
  partial_train_targets <- train_targets[-val_indices]
  
  model <- build_model()
  
  history <- model %>% fit(
    partial_train_data, partial_train_targets,
    validation_data = list(val_data, val_targets),
    epochs = num_epochs, batch_size = 16, verbose = 0
  )
  mae_history <- history$metrics$val_mae
  all_mae_histories[[i]] <- mae_history
}

#Plot validation scores
all_mae_histories <- do.call(cbind, all_mae_histories)
average_mae_history <- rowMeans(all_mae_histories)
truncated_mae_history <- average_mae_history[-(1:10)]
plot(average_mae_history, xlab = "epoch", type = 'l',
     ylim = range(truncated_mae_history))

#Train final model
model <- build_model()
model %>% fit(train_data, train_targets,
              epochs = 120, batch_size = 16, verbose = 0)
result <- model %>% evaluate(test_data, test_targets)
result["mae"]

#Generate predictions on new data
predictions <- model %>% predict(test_data)

#Prediction on 1st house
predictions[1, ]
