# King County, WA Housing Data: A Multicollinearity Journey
# 20190831

library(readr)
kc_house_data <- read_csv("NLP research/blog/6-post-042219/kc_house_data.csv", 
                          col_types = cols(condition = col_skip(), 
                                           date = col_skip(), grade = col_skip(), 
                                           id = col_skip(), view = col_skip(), 
                                           waterfront = col_skip(), yr_built = col_skip(), 
                                           yr_renovated = col_skip(), zipcode = col_skip()))
View(kc_house_data)

# check multicollinearity
sum(kc_house_data[["sqft_living"]] != kc_house_data[["sqft_above"]] + kc_house_data[["sqft_basement"]])

# add in logprice column
kc_house_data[["logprice"]] <- log(kc_house_data[["price"]])

# do the regression
lmodel <- lm(formula = logprice ~ bedrooms + bathrooms
             + sqft_living + sqft_lot + sqft_above
             + sqft_basement + lat + long, 
     data = kc_house_data)
summary(lmodel)