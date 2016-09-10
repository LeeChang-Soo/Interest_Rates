## Use Quandl to obtain market data
library(Quandl)
library(neuralnet)
library(ggplot2)
library(reshape2)

start_date <- "1982-01-04"
end_date <- "2015-10-09"

fed.rate <- Quandl("FRED/DFF", trim_start=start_date, trim_end=end_date)
tbill.3 <- Quandl("FRED/DGS3MO", trim_start=start_date, trim_end=end_date)
tbill.6 <- Quandl("FRED/DGS6MO", trim_start=start_date, trim_end=end_date)
tbill.12 <- Quandl("FRED/DGS1", trim_start=start_date, trim_end=end_date)

data <- merge(fed.rate, tbill.3, by="DATE")
data <- merge(data, tbill.6, by="DATE")
data <- merge(data, tbill.12, by="DATE")

colnames(data) <- c("Date", "fed.rate", "tbill.3", "tbill.6", "tbill.12")

m <- dim(data)[1] ## Measure the length of the file to split into separate training and actual datasets

## Randomly sample one-third of the dataset for training purposes
set.seed(3302016) ## Set the random number generator to ensure I get the same sample
val <- sample(1:m, size = round(m/3), replace = FALSE, prob = rep(1/m, m)) 
data.train <- data[-val,] ## Assign training dataset
data.test <- data[val,] ## Assign test dataset

y_test <- data.test['tbill.12']  ## Cleave off 10 year from test data for prediction error calculation
x_test <- data.test[, 2:4]  ## Cleave off 3 and 6 months from test data to pass through neural net.

ann.1 <- neuralnet(tbill.12 ~ tbill.6 + tbill.3 + fed.rate, data=data.train, 
                   threshold=0.5, lifesign="full", hidden=5, rep=1, linear.output=TRUE, stepmax=1000000,
                   algorithm = "rprop+", err.fct="sse", act.fct="logistic", learningrate=2)  

ann.results <- compute(ann.1, x_test) ## Run the 3 and 6 month test data through the ANN to get a prediction for the 10 year rate.

error <- (ann.results$net.result - y_test)*100  ## Calculate the prediction error in basis points.

results <- data.frame(ann.results$net.result)

qplot(y_test, results)


##plot(error, type="l", col="darkblue", ylim=c(-700,700), xlab="Time", ylab="Error (Basis Points)", main="ANN Prediction Error")