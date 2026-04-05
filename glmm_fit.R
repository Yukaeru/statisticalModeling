
library(glmmML)
df <- read.csv("data.csv")
model <- glmmML(cbind(y, N-y) ~ x, cluster=df$id, data=df, family=binomial)
beta1 <- model$coefficients[1]
beta2 <- model$coefficients[2]
s     <- model$sigma
write.csv(data.frame(beta1=beta1, beta2=beta2, sigma=s), "glmm_params.csv", row.names=FALSE)
