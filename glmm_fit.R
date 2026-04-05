
library(lme4)
df <- read.csv("data.csv")
model <- glmer(cbind(y, N-y) ~ x + (1|id), data=df, family=binomial)
params <- c(fixef(model), sigma=as.numeric(VarCorr(model)$id)^0.5)
write.csv(data.frame(t(params)), "glmm_params.csv", row.names=FALSE)
