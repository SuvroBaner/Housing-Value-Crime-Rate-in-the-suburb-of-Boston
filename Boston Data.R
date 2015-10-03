################## Housing Values in the suburb of Boston ########################

### Fit classification models in order to predict whether a given suburb has a crime rate above or below the median ###

install.packages("MASS")  # to access the Boston data set.
install.packages("leaps") # for best subset selction process
install.packages("glmnet") # for ridge and lasso regression
install.packages("pls")  # for pcr and pls
install.packages("tree") # for regression & classification trees
install.packages("randomForest")  # for random forest and bagging of the trees
install.packages("gbm")  # to fit boosted regression trees.

library(leaps)
library(MASS)
library(glmnet)
library(pls)
library(tree)
library(randomForest)
library(gbm)

names(Boston)
dim(Boston)
summary(Boston)
fix(Boston)
attach(Boston)

crim01 = rep(0, length(crim))
crim01[(crim > median(crim))] = 1
Boston = data.frame(Boston, crim01)

train = 1:(dim(Boston)[1]/2)
test = (dim(Boston)[1]/2 + 1):dim(Boston)[1]

Boston.train = Boston[train, ]
Boston.test = Boston[test, ]
crim01.test = Boston.test$crim01

###### Logistic Regression ######

glm.fit = glm(crim01 ~ . - crim, data = Boston, family = binomial, subset = train)
summary(glm.fit)

glm.probs = predict(glm.fit, Boston.test, type = "response")

glm.pred = rep(0, length(glm.probs))
glm.pred[(glm.probs > 0.5)] = 1
table(glm.pred, crim01.test)
mean(glm.pred != crim01.test)

# The test error rate from the Logistic model is 18.18 %

###### Linear Discriminant Analysis ######

lda.fit = lda(crim01 ~ . -crim, data = Boston, subset = train)
lda.fit

lda.pred = predict(lda.fit, Boston.test)
names(lda.pred)
lda.class = lda.pred$class
table(lda.class, crim01.test)
mean(lda.class != crim01.test)

# The test error rate from LDA is 13.4 %.

############### Now lets predict the per capita crime rate in the Boston data set.

#### Best subset selection method, and checking for Cp, BIC, AdjR2 to minimize the test error ###

Boston = Boston[, -15]  # removing crim01

regfit.full = regsubsets(crim ~ . , data = Boston, nvmax = 13)
reg.summary = summary(regfit.full)

par(mfrow = c(2, 2))

plot(reg.summary$rss, ylab = "RSS", type = "l")

plot(reg.summary$adjr2, ylab = "Adjusted Rsq", type = "l")
points(which.max(reg.summary$adjr2), reg.summary$adjr2[which.max(reg.summary$adjr2)], col = "red", cex = 2, pch = 20)

plot(reg.summary$cp, ylab = "Cp", type = "l")
points(which.min(reg.summary$cp), reg.summary$cp[which.min(reg.summary$cp)], col = "red", cex = 2, pch = 20)

plot(reg.summary$bic, ylab = "BIC", type = "l")
points(which.min(reg.summary$bic), reg.summary$bic[which.min(reg.summary$bic)], col = "red", cex = 2, pch = 20)

# As per AdjR2 it is a 9 variable model
# As per Cp it is a 8 variable model
# As per BIC it is a 3 variable model

# Now let's use the in-built plot functionality of regsubsets()

par(mfrow = c(2, 2))
plot(regfit.full, scale = "r2")
plot(regfit.full, scale = "adjr2")
plot(regfit.full, scale = "Cp")
plot(regfit.full, scale = "bic") 

# As per r2 : all 13 predictors are statistically significant
# As per adjr2 : zn, indus, nox, dis, rad, ptratio, black, lstat , medv  (9 predictors)
# As per Cp : zn, nox, dis, rad, ptratio, black, lstat , medv  (8 predictors)
# As per BIC : rad, black, lstat  (3 predictors)

coef(regfit.full, 3)

#### Let's use forward and backward stepwise selection using subset apprraoch ###

regfit.fwd = regsubsets(crim ~ . , data = Boston, nvmax = 13, method = "forward")

regfit.bwd = regsubsets(crim ~ . , data = Boston, nvmax = 13, method = "backward")

par(mfrow = c(2, 2))
plot(regfit.fwd, scale = "r2")
plot(regfit.fwd, scale = "adjr2")
plot(regfit.fwd, scale = "Cp")
plot(regfit.fwd, scale = "bic")

# The results are identical using Forward stepwise selection

par(mfrow = c(2, 2))
plot(regfit.bwd, scale = "r2")
plot(regfit.bwd, scale = "adjr2")
plot(regfit.bwd, scale = "Cp")
plot(regfit.bwd, scale = "bic")

# Using Backward selection r2, adjr2 and Cp are identical. But BIC gives a different result of best 4 predictor model
# zn, dis, rad, medv  ( which is quite interesting, we'll investigate it further)

#### Now, perform Best Subset selection and choose the best model using Validation set and Cross-validation approach (directly estimate test MSE)###

set.seed(1)

train = sample(c(TRUE, FALSE), nrow(Boston), rep = TRUE)
test = (!train)

regfit.train = regsubsets(crim ~ . , data = Boston[train,], nvmax = 13)

test.mat = model.matrix(crim ~ . , data = Boston[test,])

val.errors = rep(NA, 13)  # validation set errors

for(i in 1:13)
{
	coefi = coef(regfit.train, id = i)
	pred = test.mat[, names(coefi)]%*% coefi
	val.errors[i] = mean((crim[test] - pred)^2)
}

val.errors
which.min(val.errors)
plot(val.errors, ylab = "RSS", type = "l")
points(which.min(val.errors), val.errors[which.min(val.errors)], col = "red", cex = 2, pch = 20)

# The best subset model with validation set approach is 5 predictor models

# Now lets do it for the entire data set to get a better estimate of the coeficients

regfit.best = regsubsets(crim ~ . , data = Boston, nvmax = 13)
coef(regfit.best, which.min(val.errors))

#(Intercept)           zn          dis          rad        black         medv 
# 7.919932813  0.051799071 -0.672189237  0.472305624 -0.008211013 -0.174218506 


### Best Subset selection and estimating test MSE using k - fold cross validation approach

k = 10
set.seed(3)
folds = sample(1:k, nrow(Boston), replace = TRUE)
cv.errors = matrix(NA, k, 13, dimnames = list(NULL, paste(1:13)))

predict.regsubsets = function(object, newdata, id,...)
{
	form = as.formula(object$call[[2]])
	mat = model.matrix(form, newdata)
	coefi = coef(object, id=id)
	xvars = names(coefi)
	mat[, xvars]%*%coefi
}


# the elements of folds that equal j are in the test set
for(j in 1:k)
{
	train.fit = regsubsets(crim ~ . , data = Boston[folds != j,], nvmax = 13)
	for(i in 1:13)
	{
		pred = predict(train.fit, Boston[folds == j,], id = i)
		cv.errors[j, i] = mean((crim[folds == j] - pred)^2)
	}	
}


# Note if you do not want to use the userdefined function, you can use the following code.
#set.seed(2)
#Boston.model.matrix = model.matrix(crim ~ . , data = Boston)
#k = 10
#folds = sample(1:k, nrow(Boston), replace = TRUE)
#cv.errors = matrix(NA, k, 13)
#
#for (j in 1:k)
#{
#  regfit.train = regsubsets(crim ~ . , data = Boston[folds != j, ], nvmax = 13)
#  
#  for (i in 1:13)
#  {
#    coefi = coef(regfit.train, id = i)
#    pred = Boston.model.matrix[folds == j, names(coefi)]%*%coefi
#    cv.errors[j, i] = mean((pred - crim[folds == j])^2)
#  }
#}

# This is a 10 x 13 matrix, of which the (i, j) th element corresponds to the test MSE for the ith cross-validation fold for the best j-th model.

mean.cv.errors = apply(cv.errors, 2, mean)
mean.cv.errors

par(mfrow = c(1, 1))
plot(mean.cv.errors, type = "b")
points(which.min(mean.cv.errors), mean.cv.errors[which.min(mean.cv.errors)], col = "red", cex = 2, pch = 20)

# Although the minimum value is 12 where MSE is 42.68571
# The value of index = 9 where MSE is 42.72009 which is very close.
# So, we will take the optimum model to be 9 variable model

# Let's run on the entire data set with this model specifiction and find the estimated coefficients.

reg.best = regsubsets(crim ~ . , data = Boston, nvmax = 9)
coef(reg.best, 9)
#Intercept)            zn         indus           nox           dis 
#19.124636156   0.042788127  -0.099385948 -10.466490364  -1.002597606 
#         rad       ptratio         black         lstat          medv 
# 0.539503547  -0.270835584  -0.008003761   0.117805932  -0.180593877 


## This is an interesting result and it also coincides with the AdjR2 result for estimating the test MSE.
## We'll move on and do other genere of model fits.

### Lets's do a ridge-regression on this data set 

set.seed(11)

train.mat = model.matrix(crim ~ . ,data = Boston[train,])
test.mat = model.matrix(crim ~ . , data = Boston[test,])
crim.train = Boston[train,]$crim
crim.test = Boston[test,]$crim

grid = 10^seq(10, -2, length = 100)

ridge.mod = cv.glmnet(train.mat, crim.train, alpha = 0, lambda = grid, thresh = 1e-12)
plot(ridge.mod)
bestlam = ridge.mod$lambda.min
bestlam

ridge.pred = predict(ridge.mod, s = bestlam, newx = test.mat)
mean((crim.test - ridge.pred)^2)

x.mat = model.matrix(crim ~ . , data = Boston)
y.mat = Boston$crim

ridge.best = glmnet(x.mat, y.mat, alpha = 0, lambda = bestlam)
coef(ridge.best)

#(Intercept)  9.455718029
#(Intercept)  .          
#zn           0.033587825
#indus       -0.082350022
#chas        -0.737483117
#nox         -5.640774955
#rm           0.338830981
#age          0.001957069
#dis         -0.716746627
#rad          0.430595489
#tax          0.003119667
#ptratio     -0.142911948
#black       -0.008440088
#lstat        0.142079579
#medv        -0.142270339

### Let's do a Lasso 

lasso.mod = cv.glmnet(train.mat, crim.train, alpha = 1, lambda = grid, thresh = 1e-12)
plot(lasso.mod)
bestlam = lasso.mod$lambda.min
bestlam

lasso.pred = predict(lasso.mod, s = bestlam, newx = test.mat)
mean((crim.test - lasso.pred)^2)

lasso.best = glmnet(x.mat, y.mat, alpha = 1, lambda = bestlam)
coef(lasso.best)

#(Intercept)  0.983690160
#(Intercept)  .          
#zn           .          
#indus        .          
#chas         .          
#nox          .          
#rm           .          
#age          .          
#dis         -0.035172876
#rad          0.458547206
#tax          .          
#ptratio      .          
#black       -0.005869646
#lstat        0.126149001
#medv        -0.049623596

### Now let's try Principal Component Regression 

set.seed(4)
pcr.fit = pcr(crim ~ . , data = Boston, subset = train, scale = TRUE, validation = "CV")
summary(pcr.fit)

validationplot(pcr.fit, val.type = "MSEP")

# The lowest cross validation error occurs when M is about 8.
# Let's compute the test MSE.

pcr.pred = predict(pcr.fit, Boston[test,], ncomp = 8)
mean((pcr.pred - crim.test)^2)

pcr.best = pcr(crim ~ . , data = Boston, scale = TRUE, ncomp = 8)
summary(pcr.fit)


################################### Regression Problem ########################################

#### Predict the median value of the owner occupied homes (in $1,000's) ####

### Fitting RegressionTrees ###
attach(Boston)

summary(Boston)

set.seed(10)
train = sample(1:nrow(Boston), nrow(Boston)/2)
boston.test = Boston[-train, "medv"]

tree.boston = tree(medv ~ . , data = Boston, subset = train)
summary(tree.boston)

# Variables actually used in tree construction:
# [1] "lstat"   "rm"      "dis"     "crim"    "ptratio"
# Number of terminal nodes:  9 
# Residual mean deviance:  10.99  (sum of squared)

plot(tree.boston)
text(tree.boston, pretty = 0)

## The variable lstat measures the percentage of individuals with lower socioeconomic status.
## The tree indicates that lower values of lstat corresponds to more expensive houses.
## The tree predicts a median house price of $ 46,400 for larger homes in suburbs in which the residents 
## have high socioeconomic status (rm >= 7.437 and lstat < 9.635)


pred.tree = predict(tree.boston, Boston[-train, ])
mean((pred.tree - boston.test)^2)

# The test set MSE is 22.74

# Now we will use cv.tree() to see whether pruning the tree will improve the performance.

cv.boston = cv.tree(tree.boston)
plot(cv.boston$size, cv.boston$dev, type = 'b')

# Let's take the value of terminal node to be 6 which has relatively low cross-validation error.

prune.boston = prune.tree(tree.boston, best = 6)
summary(prune.boston)

# Variables actually used in tree construction:
# [1] "lstat" "rm"    "dis"   "crim" 
# Number of terminal nodes:  6 
# Residual mean deviance:  14.39
# As this deviance is more than the unprunned tree, we would go with the original (unprunned tree)

yhat = predict(prune.boston, newdata = Boston[-train,])
plot(yhat, boston.test)
abline(0, 1)
mean((yhat - boston.test)^2)

# In other words, the test set MSE associated with the regression tree is 28.39
# The square root of the MSE is therefore around 4.768, indicating that this model leads to test predictions
# that are within around $ 5,375 of the true median home value of the suburb.

################# Bagging and Random Forests ############

############## Bagging process (it is a type of boot strapping)

# To apply bagging to regression trees, we simply construct 'B' regression trees using 'B' bootstrapped training sets,
# and average the resulting predictors. The trees are grown deep and not pruned. Averaging 'B' trees reduces the variance.

set.seed(11)
bag.boston = randomForest(medv ~ ., data = Boston, subset = train, mtry = 13, importance = TRUE)

# here, argument mtry = 13 indicates that all 13 predictors should be considered for each split of the tree.
# Bagging does a bootstrapping (adding many)using all the predictors , at all times.

bag.boston

# Type of random forest: regression
#                     Number of trees: 500
# No. of variables tried at each split: 13

# Mean of squared residuals: 12.26502
#                    % Var explained: 85.43

yhat.bag = predict(bag.boston, newdata = Boston[-train, ])
plot(yhat.bag, boston.test)
abline(0,1)
mean((yhat.bag - boston.test)^2)


# The test set MSE associated with the bagged regression tree is 12.2, almost half that obtained using an 
# optimally-prunned single tree.

# We could change the number of trees grown by randomForest() using the ntree argument.

bag.boston = randomForest(medv ~ . , data = Boston, subset = train, mtry = 13, ntree = 25)
yhat.bag = predict(bag.boston, newdata = Boston[-train, ])
mean((yhat.bag - boston.test)^2)

# The test MSE is 12.39

########### Random Forests:
# For Random forest we use a smaller value of mtry argument. By default randomForest() uses p/3 variables
# when building a random forest of regression trees, and sqrt(p) variables when building a random forest of classification trees.

# Here we use, mtry = 6

set.seed(1)
rf.boston = randomForest(medv ~ . , data = Boston, subset = train, mtry = 6, importance = TRUE)
yhat.rf = predict(rf.boston, newdata = Boston[-train, ])
mean((yhat.rf - boston.test)^2)

# The test MSE is 11.08, this indicates that random forests yielded an improvement over the bagging in this case.
importance(rf.boston) # using importance(), we can view the importance of each variable.

# Two measures of variable importance are reported. 
# The former is based upon the mean decrease of accuracy in predictions on the out of bag samples when a given variable is excluded from the model.
# The later is a measure of the total decrease in node impurity that results from splits over that variable, averaged over all trees.
# In the case of regression trees, the node impurity is measured by the training RSS, and for classification trees by deviance.
# This plot can be using below.


varImpPlot(rf.boston)

## The results indicate that across all of the trees considered in the random forest, the wealth level of the community (lstat)
## and the house size (rm) are by far the two most important variables.

##################################### Boosting of regression trees #################################

# Here we use the gbm package, and within it the gbm() to fit boosted regression trees to the Boston data set.
# We run gbm() with the option, distribution = "gaussian" since this is a regression problem.
# we would use distribution = "bernoulli" if it's a classification problem.
# The argument n.trees = 5000 indicates that we want 5000 trees, and the option 
# interaction.depth = 4 limits the depth of each tree.

set.seed(12)

boost.boston = gbm(medv ~ . , data = Boston[train, ], distribution = "gaussian", n.trees = 5000, interaction.depth = 4)
summary(boost.boston)  # it reports the relative influence statistics.

# By far, lstat and rm are the most important variables.
# We can also produce partial dependence plots for these two variables. These plots illustrate
# the marginal effect of the selected variables on the response after integrating out the other variables.
# In this case, as we might expect, median house prices are increasing with rm and decreasing with lstat.

par(mfrow = c(1, 2))
plot(boost.boston, i = "rm")
plot(boost.boston, i = "lstat")

yhat.boost = predict(boost.boston, newdata = Boston[-train, ], n.trees = 5000)
mean((yhat.boost - boston.test)^2)

# The test MSE obtained is 11.8, similar to the test MSE for random forests and superior to that for bagging.
# If we want to, we can perform boosting with different value of the shrinkage parameter lambda

boost.boston = gbm(medv ~ . , data = Boston[train, ], distribution = "gaussian", n.trees = 5000, interaction.depth = 4, shrinkage = 0.2, verbose = F)
yhat.boost = predict(boost.boston, newdata = Boston[-train, ], n.trees = 5000)
mean((yhat.boost - boston.test)^2)

# Here the test MSE is 13.5
