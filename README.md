# Holiday-Package-Prediction
A model for predict users who have the potential to buy Wellness Tourism Package.

# Dataset Description
This data is offer data on holiday packages originating from a travel company and plans to expand its business through the latest package offerings, namely the Wellness Tourism Package with the target of whether the holiday package is taken or not(ProdTaken).

Source: ML Repo > <a href="https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction">Holiday_Package_Prediction</a>

| Columns | Description | Type | 
| :- | :- | :- |
| CustomerID | Unique customers ID | | 
| ProdTaken | does the customers take a vacation package? (0: No, 1: Yes) | Target Variable | 
| Age  | Age of customer | Numeric | 
| TypeofContact  | How customer was contacted (Company Invited or Self Inquiry) | Categorical | 
| CityTier | City tier depends on the development of a city, population, facilities, and living standards. The categories are ordered i.e. | Categorical | 
| DurationOfPitch | Duration of the pitch by a salesperson to the customer | Numeric | 
| Occupation | Occupation of customer | Categorical | 
| Gender | Gender of customer | Categorical | 
| NumberOfPersonVisiting | Total number of persons planning to take the trip with the customer | Numeric | 
| NumberOfFollowups | Total number of follow-ups has been done by the salesperson after the sales pitch | Numeric | 
| ProductPitched | Product pitched by the salesperson | Categorical | 
| PreferredPropertyStar | Preferred hotel property rating by customer | Categorical |
| MaritalStatus | Marital status of customer | Categorical | 
| NumberOfTrips | Average number of trips in a year by customer | Numeric | 
| Passport | The customer has a passport or not (0: No, 1: Yes) | Boolean | 
| PitchSatisfactionScore | Sales pitch satisfaction score | Categorical | 
| OwnCar | Whether the customers own a car or not (0: No, 1: Yes) | Boolean | 
| NumberOfChildrenVisiting | Total number of children with age less than 5 planning to take the trip with the customer | Numeric | 
| Designation | Designation of the customer in the current organization | Categorical | 
| MonthlyIncome| Gross monthly income of the customer | Numeric | 

# Background and Objective
## The problem behind this analysis.
The company Trips&Travel.com wants to launch a new vacation package called the Wellness Tourism Package. Unfortunately, they have a problem with promotion. They contact random customers to offer vacation packages so it is not efficient. Based on last year’s data, Trips&Travel.com has 4888 customers. However, only 18% of the total customers purchased vacation packages.

## The importance of this analysis carried out.
This analysis is important in order that the promotion can run efficiently to optimize the use of costs, with the following objectives that can be broken down into 4:
1. Compare the performance of each algorithm as follows:

    Logistic Regression,
    Decision Tree,
    Random Forest,
    K-Nearest Neighbors,
    AdaBoost,
    XGBoost.
2. Decide which algorithm we will use based on the best performance on the chosen metrics.
3. Percentage of users who bought vacation packages and calculation of Customer Acquisition Cost (CAC).
4. Analyze what kind of customers are more likely to take vacation packages.‍

# Scope of problem
The scope of the problem is to be solved by measuring the following business metrics:
1. Percentage of users who bought vacation packages
2. Customer Acquisition Cost (CAC)
3. Revenue
The output of machine learning to be created is to predict users who have the potential to buy vacation packages.

# Data and Assumptions‍
This dataset contains 4K (4888) rows and 20 columns. Amongst the columns, we have the target variable is named ‘ProdTaken’, which contains ‘Do the customers take a vacation package?’ (0: No, 1: Yes). This dataset is a mix of numerical and categorical variables. The target variable is in categorical data type and it’s already converted to boolean type. All features are used in machine learning except for the customer ID because previously we have tried to use features consisting of passport, age, monthly income, type of contact, own car, city tier, occupation, marital status, number of follow-ups, number of trips and product pitched. We decided to use those features based on the results of the bivariate and multivariate analyses that we have done before, but when we try to add all the features we get better performance of results.

# Data Analysis
## Univariate

![alt text](https://github.com/ayodhyaGA/Holiday-Package-Prediction/blob/master/num%20histplot.png) 
Observation results:
- Column ‘Age’ has a distribution that is close to normal or symmetrical, it can be seen from the picture and the skew value is close to 0.5 (0.41).
- Columns ‘Duration Of Pitch’ and ‘Monthly Income’ have a positive skew (skew to the right) i.e. more concentrated on the left and a long tail on the right side, the skew results also show their respective values are 1.84 and 1.7, respectively.
- The ‘Monthly Income’ column allows for normalization because the variation in income is quite large for the numerical column with a range variation is large.
- Other columns are not given much attention because among them are ordinal and nominal type columns whose format is made in numeric form so it still needs to be done at the preprocessing stage for Feature Encoding.

![alt text](https://github.com/ayodhyaGA/Holiday-Package-Prediction/blob/master/num%20boxplot.png)
On the upper limits: DurationOfPitch, NumberOfPersonVisiting, and NumberOfTrips.
Below the bottom and upper limits: NumberOfFollowups and MonthlyIncomemeric Fitur.

![alt text](https://github.com/ayodhyaGA/Holiday-Package-Prediction/blob/master/univariat%20plot.jpg)
Observation results:
- The Self Inquiry category of the ‘Type of Contact’ column has more proportion than the Company Invited category.
- In the ‘Occupation’ column, most users come from the Salaried category, while in the FreeLancer category there are almost none.
- Users in the ‘Gender’ column are dominated by Male.
- In the ‘Product Pitched’ column the most categories are Basic, then the number is decreasing in the Deluxe to King categories.
- In the ‘Marital Status’ column, the number of users in the Married category is the most compared to other categories.
- Users who are included in the Executive category in the ‘Designation’ column rank the most followed by the Manager, Senior Manager, AVP, and VP categories.

## Bivariate
![alt text](https://github.com/ayodhyaGA/Holiday-Package-Prediction/blob/master/numerik%20plot.jpg)
The age distribution of customers who pick up or do not pick up the package has almost the same symmetric distribution. While the duration of the sales force making offers there is also no significant difference in the distribution, which is symmetrical to the right. It same for the monthly income column, the slope distribution is more symmetrical to the right for customers who take the package or not.

![alt text](https://github.com/ayodhyaGA/Holiday-Package-Prediction/blob/master/count%20plot.jpg)
- The result of the frequency that has more customers do not take vacation packages, is in accordance with the problem statement which explains that only 18% of customers take vacation packages for offers that are made randomly so it is not appropriate in marketing.
- However, the type of customer contact in making an offer with a larger ratio is the company invited.
- Type of work type has a higher ratio in large businesses, the type of product has a higher ratio than the basic product and a larger designation ratio in executive positions.
From the results of descriptive statistical observations of numerical features, we also find:
- The age of customers who took the package or not has a median value that was not much different, but slightly more age variation was found among those who did not take the vacation package.
- The duration of the officer in providing the offer is longer found in customers who take the vacation package, it was described either through the median or average value, although the maximum duration is greater for those who do not take the vacation package, this outlier needs to be re-examined. In addition, the duration of offering at the 75% percentile has two times more customers took the vacation package.
- The monthly income range is greater for customers are not take vacation packages, this can be seen through the Peak-to-peak (maximum-minimum) or PTP and standard deviation values, but both the mean and median values do not look much different.
The number of trips there is no significant difference between those who take the package or not.

## Multivariate
![alt text](https://github.com/ayodhyaGA/Holiday-Package-Prediction/blob/master/heatmap.jpg)
- No feature has a strong correlation with the target (>0.7).
- There are several features to consider, namely 'Passport' (0.23), 'CityTier' (0.11), 'NumberOfFollowups' (0.13), 'Age' (-0.18), and 'MonthlyIncome' (-0.14) because it has a higher correlation value than other features.
- The correlation between features is quite large in the 'MonthlyIncome' feature with 'Age' which has a correlation value of 0.45. The feature 'NumberOfChildrenVisiting' with 'NumberOfPersonVisiting' has a correlation value of 0.61. However, both of them still cannot be said to be redundant because the correlation value is still < 0.7.

The process of selecting the features and performance of each model with predetermined metrics.
The metrics used are Recall as the primary metrics but we also set a threshold for precision >= 0.5. We chose to consider these matrices because we don’t want a large false negative value (customers who actually want to buy vacation packages but are thought to not want to buy).
# Model Analysis
From a total of 246 customers who took vacation packages, as many as 199 could be predicted correctly. we have reached the expected performance 80%. 

# Evaluation
- Based on the recall score, XGBoost has the best performance compared to other algorithms. The results from the test train are slightly higher than the test results but are still in the range of ~0.80–0.87.
- Performance results based on the precision score on XGBoost are also at >=0.50 according to the threshold that we have set, namely with a score of 0.67.
- When viewed from the accuracy of the XGBoost modeling, the score is still quite high, namely 0.72
- The highest precision and accuracy values are obtained by the KNearest Neighbors model, but the results of this score are still overfit than the XGBoost model even though hyperparameter tuning has been carried out to prevent overfit tendencies.
- Time in predicting scores, the XGBoost model gets the 3rd fastest position out of all models, while of all time modeling the fastest in predicting scores is found in Decision Tree modeling.
- Hyperparameter tuning performed in XGBoost includes: cross-validation, max_depth, gamma, learning_rate, tree_method, colsample_bytree, subsample, early_stopping_rounds.

# Confusion Matrix
After we found that the XGBoost score had the best performance seen from the recall results. We also saw how the Confusion Matrix results from the XGBoost modeling. In the following figure, it can be seen that from a total of 246 customers who purchased vacation packages, 199 were predicted correctly or we managed to get an expected performance of 80%.

# Shap Value
In addition to being seen from the important feature, we get features from the XGBoost model that have an important role through shape value. This is in accordance with the results of the important feature that the features consisting of Passport, Designation, Product Pitched, Single Marital Status, and City Tier are the most influential features. We also look at the shape value results to see how the direction of the relationship is with the target feature as shown in the following figure.
![alt text](https://github.com/ayodhyaGA/Holiday-Package-Prediction/blob/master/shapvalue.png)

From the results of the SHAP value, it can be seen that:

- Passport is the most influential feature, people who take vacation packages are people who have more passports, this can be seen through the shape value in the range 0.1–0.2
- Customers with basic positions have a high probability of taking vacation packages. with a range of shape values 0–0.1
- Customers who take vacation packages are also those who take basic and standard packages.
- Single customers are more likely to take vacation packages.

# Conclusion
The final model we chose was XGBoost with 80% performance from the recall matrix or it can be said that this model can predict the number of customers who buy vacation packages by 80%. In addition, we were able to reduce CAC previously from the marketing side by contacting customers randomly to make offers to spend CAC of $5.31 after implementing a machine learning model so that offers made to customers are more effective, only spending CAC of $1.25. From the results of the CAC calculation, the company can get a revenue difference of $3,726 after applying the machine learning model. Important features to consider in bidding to customers from modeling results using XGBoost are Passport, Designation, Product Pitched, Single Marital Status, and City Tier.

# Business Recommendations Suggestion
## Action points (High Potential Customers)
Offering vacation packages to domestic & international destinations for customers who have passports.
Offering Wellness packages at low prices, such as Basic & Standard to customers who have never purchased a vacation package and for those who have Executive & Manager positions.
Intensifying promotions to customers from big & medium cities.
Single offering customers, it is advisable to promote holiday packages outside of major holidays.
## Action points (Low Potential Customers)
Offering Wellness Packages for domestic destinations for customers who do not have a passport.
Offers Wellness Packages with a medium to high price range for those who have at least senior manager positions.
Customers who are married will get a Wellness Package promotion on major holidays (such as New Year, etc.).
