#Implement linear classifier from scratch
num_samples_per_class <- 1000
Sigma <- rbind(c(1, 0.5),
               c(0.5, 1))

negative_samples <-
  MASS::mvrnorm(n = num_samples_per_class,
                mu = c(0, 3), Sigma = Sigma)
positive_samples <-
  MASS::mvrnorm(n = num_samples_per_class,
                mu = c(3, 0), Sigma = Sigma)

#Stack the two classes into an array with shape (2000, 2)
inputs <- rbind(negative_samples, positive_samples)

#Generate the corresponding targets (0 and 1)
targets <- rbind(array(0, dim = c(num_samples_per_class, 1)),
                 array(1, dim = c(num_samples_per_class, 1)))

#Plot the two point classes
plot(x = inputs[, 1], y = inputs[, 2],
     col = ifelse(targets[, 1] == 0, "purple", "green"))

#Create the linear classifier variables
input_dim <- 2
output_dim <- 1
W <- tf$Variable(initial_value =
                   tf$random$uniform(shape(input_dim, output_dim)))
b <- tf$Variable(initial_value = tf$zeros(shape(output_dim)))

#Forward pass function
model <- function(inputs)
  tf$matmul(inputs, W) + b

#The mean squared error loss function
square_loss <- function(targets, predictions) {
  per_sample_losses <- (targets - predictions)^2
  mean(per_sample_losses)
}

#Training step function
learning_rate <- 0.1

training_step <- function(inputs, targets) {
  with(tf$GradientTape() %as% tape, {
    predictions <- model(inputs)
    loss <- square_loss(predictions, targets)
  })
  grad_loss_wrt <- tape$gradient(loss, list(W = W, b = b))
  W$assign_sub(grad_loss_wrt$W * learning_rate)
  b$assign_sub(grad_loss_wrt$b * learning_rate)
  loss
}

#The batch training loop
inputs <- as_tensor(inputs, dtype = "float32")
for (step in seq(40)) {
  loss <- training_step(inputs, targets)
  cat(sprintf("Loss at step %s: %.4f\n", step, loss))
}

predictions <- model(inputs)

inputs <- as.array(inputs)
predictions <- as.array(predictions)
plot(inputs[, 1], inputs[, 2],
     col = ifelse(predictions[, 1] <= 0.5, "purple", "green"))

slope <- -W[1, ] / W[2, ]
intercept <- (0.5 - b) / W[2, ]
abline(as.array(intercept), as.array(slope), col = "red")





