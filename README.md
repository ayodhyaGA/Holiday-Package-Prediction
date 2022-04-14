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

# Objective
Create a model for predict users who have the potential to buy Wellness Tourism Package.

The main objective can be broken down into 3:
1. Compare the performance between each algorithm as follows:
    - Logistic Regression,
    - Decision Tree,
    - Random Forest,
    - K-Nearest Neighbors,
    - AdaBoost,
    - XGBoost.
2. Decide which algorithm that we will use based on the best performance on the chosen metrics.
3. Percentage of users who bought vacation packages.
4. Analyze what kind of customers are more likely to take vacation packages.

# Insights From EDA
**Univariate**
- Column 'Age' has a distribution that is close to normal or symmetrical, it can be seen from the picture and the skew value is close to 0.5 (0.41).
- Columns 'DurationOfPitch' and 'MonthlyIncome' have a positive skew (skew to the right) i.e. more concentrated on the left and a long tail on the right side, the skew results also show their respective values are 1.84 and 1.7, respectively.
- Other columns are not given much attention because among them are ordinal and nominal type columns whose format is made in numeric form so that it still needs to be done at the preprocessing stage for Feature Encoding.
- The Self Inquiry category of the 'TypeofContact' column has more proportion than the Company Invited category.
- In the 'Occupation' column, most users come from the Salaried category, while in the Free Lancer category there are almost none.
- Users in the 'Gender' column are dominated by Male.
- In the 'ProductPitched' column the most categories are Basic, then the number is decreasing in the Deluxe to King categories.
- In the 'MaritalStatus' column, the number of users in the Married category is the most compared to other categories.
- Users who are included in the Executive category in the 'Designation' column ranks the most followed by the Manager, Senior Manager, AVP, and VP categories.

**Bivariate**
- The age distribution of customers who pick up or no pick up the package has almost the same symmetric distribution. While the duration of the sales force making offers there is also no significant difference of the distribution, which is symmetrical to the right. It same for the monthly income column, the slope distribution is more symmetrical to the right for customers who take the package or not.
- The result of the frequency that has more customers do not take vacation packages, this is in accordance with the problem statement which explains that only 18% of customers take vacation packages for offers that are made randomly so that it is not appropriate in marketing.
- However, the type of customer contact in making an offer with a larger ratio is the company invited.
- The age of customers who took the package or did not have a median value that was not much different, but slightly more of age variation was found among those who did not take the vacation package.
- The duration of the officer in providing the offer is longer found in customers who take the vacation package, it was described either through the median or average value, although the maximum duration is greater for those who do not take the vacation package, this outlier needs to be re-examined. In addition, the duration of offering at the 75% percentile has two times more customers took the vacation package.
- The monthly income range is greater for customers are not take vacation package, this can be seen through the ptp and std values, but both the mean and median values do not look much different.
- the number of trips there is no significant difference between those who take the package or not.

**Multivariate**
- No feature has a strong correlation with the target (>0.7).
- There are several features to consider, namely 'Passport' (0.23), 'CityTier' (0.11), 'NumberOfFollowups' (0.13), 'Age' (-0.18), and 'MonthlyIncome' (-0.14) because it has a higher correlation value than other features.
- The correlation between features is quite large in the 'MonthlyIncome' feature with 'Age' which has a correlation value of 0.45. The feature 'NumberOfChildrenVisiting' with 'NumberOfPersonVisiting' has a correlation value of 0.61. However, both of them still cannot be said to be redundant because the correlation value is still < 0.7.


# Model Analysis
From a total of 246 customers who took vacation packages, as many as 199 could be predicted correctly. we have reached the expected performance 80%. 

**Evaluation**
- Based on the recall score, XGBoost has the best performance compared to other algorithms. The results from the test train are slightly higher than the test results but are still in the range of ~0.80 - 0.87.
- Performance results based on the precision score on XGBoost are also at >=0.50 according to the threshold that we have set, namely with a score of 0.67.
- When viewed from the accuracy of the XGBoost modeling, the score is still quite high, namely 0.72
- The highest precision and accuracy values are obtained by the KNearest Neighbors model, but the results of this score are still overfit than the XGBoost model even though hyperparameter tuning has been carried out to prevent overfit tendencies.
- Time in predicting scores, the XGBoost model gets the 3rd fastest position out of all models, while of all time modeling the fastest in predicting scores is found in Decision Tree modeling

So we decided to use the XGBoost model which has the best performance, with the result of the shap value we can see the direction of the relationship between all features and target variables as follows:
- Passport is the most influential feature, people who take vacation packages are people who have more passports, this can be seen through the shape value in the range 0.1-0.2
- Customers with basic positions have a high probability of taking vacation packages. with a range of shape values 0 - 0.1
- Customers who take vacation packages are also those who take basic and standard packages.
- Single customers are more likely to take vacation packages
