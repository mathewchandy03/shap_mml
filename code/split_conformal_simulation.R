library(MASS)
library(tidyverse)

make_psd <- function(M) {
  eig <- eigen((M + t(M))/2)
  eig$values[eig$values < 0] <- 1e-8
  M_psd <- eig$vectors %*% diag(eig$values) %*% t(eig$vectors)
  return(M_psd)
}

squared_error <- function(x, y, mu, S, d) {
  x_S <- x
  x_S[-(d * rep(S, d) + rep((-d + 1):0, each = length(S)))] <- 0
  y_preds <- predict(mu, data.frame(t(x_S)))
  (y - y_preds) ^ 2
}

shapley_value <- function(x, y, val, mu, j, p) {
  d <- length(x) / p
  value <- 0
  for (m in 0:(p-1)) {
    weight <- factorial(m) * factorial(p - m - 1) / factorial(p)
    coalitions <- combn((1:p)[-j], m)
    sse <- 0
    for (coal in 1:ncol(coalitions)) {
      S <- coalitions[,coal]
      S_with_j <- c(S, j)
      sse <- sse + val(x, y, mu, S_with_j, d) - val(x, y, mu, S, d)
    }
    value <- sse * weight
  }
  return(value)
}

set.seed(123)
p <- 10
d <- 3
B1 <- matrix(runif((p * d) ^ 2, -1, 1), p * d, p * d)
B2 <- B1 %*% t(B1)
B <- B2 / rowSums(abs(B2))
epsilon <- 0.1
n = 1000
mu <- rep(0, 30)
beta <- runif(p * d, -1, 1)
alpha <- runif(1, -1, 1)
error <- rnorm(n)

A <- matrix(0, p * d, p * d)
for (i in 1:p) {
  A1 <- matrix(runif(d ^ 2, -1, 1), d, d)
  A2 <- A1 %*% t(A1)
  j <- d * i - d + 1
  A[j:(j + d - 1), j:(j + d - 1)] <- A2 / rowSums(abs(A2))
}

Sigma <- (1 - epsilon) * A + epsilon * B
Sigma <- make_psd(Sigma)
X <- mvrnorm(n, mu, Sigma)
columns <- numeric(p * d)
for (i in 1:p) {
  j <- d * i - d + 1
  columns[j:(j + d - 1)] <- paste0(paste("X_", i, '_', sep = ''), 1:d)
}
colnames(X) <- columns

Y <- as.vector(X %*% beta) + alpha + error

alpha <- 0.20

I1 <- sample(1:n, n / 2)
I2 <- setdiff(1:n, I1)
D1 <- data.frame(Y = Y[I1], X[I1,])
mu <- lm(Y ~ ., D1)

values <- matrix(NA, nrow = n / 2, ncol = p)

count = 0
for (i in I2) {
  count = count + 1
  for (j in 1:p) {
    values[count, j] = shapley_value(X[i,], 
                                     Y[i], squared_error, mu, j, p)
  }
  print(paste('Iteration', count, 'done'))
}

saveRDS(values, file = '../data/values.RDS')

df <- as.data.frame(values)
df_long <- pivot_longer(df, cols = everything(), names_to = "Modality", values_to = "Shapley Value")
df_long$Modality <- gsub("^V", "X[", df_long$Modality)
df_long$Modality <- paste0(df_long$Modality, ']')
df_long$Modality <- factor(df_long$Modality, levels = paste0("X[", 1:10, ']'))

ggplot(df_long, aes(x = `Shapley Value`)) +
  geom_boxplot() +
  facet_wrap(~ Modality, nrow = 5, ncol = 2, scales = "fixed",
             labeller = label_parsed) +
  theme_minimal()

ggsave('../manuscript/figures/modality_attribution_boxplot.pdf')
ell <- ceiling((n / 2 + 1) * (alpha / 2))
u <- ceiling((n / 2 + 1) * (1 - alpha / 2))

shapley_intervals <- matrix(NA, nrow = p, ncol = 2)
for (j in 1:p) {
  my_values <- sort(values[,j])
  shapley_intervals[j, 1] <- my_values[ell]
  shapley_intervals[j, 2] <- my_values[u]
}

df <- data.frame(
  modality = factor(paste0("X[", 1:10, ']'), levels = paste0("X[", 1:10, ']')),
  lower = shapley_intervals[, 1],
  upper = shapley_intervals[, 2]
)

ggplot(df, aes(x = modality, ymin = lower, ymax = upper)) +
  geom_errorbar(width = 0.2) +
  labs(x = 'Modality', y = "Shapley Value 80% Interval") +
  theme_bw() + 
  scale_x_discrete(labels = parse(text = levels(df$modality))) +
  geom_hline(yintercept = 0, linetype = 'dotted', color = 'red')

ggsave('../manuscript/figures/shapley_value_intervals.pdf')
