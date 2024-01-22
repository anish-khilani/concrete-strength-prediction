# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
%matplotlib inline
import scipy.stats as stats

# %% [markdown]
# ## Data Analysis

# %%
## The Shape of the Data Frame 

dataFrame = pd.read_csv('Concrete_Strength.csv')            ## Load the Data
print('Data frame :', dataFrame.shape)                      ## Display the Shape
numericFeatures = list(dataFrame.columns.values)            ## Name of the features 
print('Features   : ', len(numericFeatures) - 1)            ## Print number of features
dataFrame.head()                                            ## Display the first 5 rows of the data frame

# %%
## Drop any duplicated rows from the data frame 

dataFrame.drop_duplicates(inplace = True)
dataFrame.reset_index(drop = True, inplace = True)

print('Data Frame shape "post duplicate removal" :', dataFrame.shape)

# %%
## Search for any missing data for the features
plt.figure(figsize = (7, 7))
sns.heatmap(dataFrame.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
plt.show()

# %%
dataFrame.hist(bins = 30, figsize = (15, 15))

# %%
dataFrame.boxplot(vert = False, grid = True, figsize = [10, 10])

# %%
## Find Correlation between features
corrMatrix = dataFrame.corr()
plt.figure(figsize = (10, 8))
sns.heatmap(corrMatrix, annot = True, annot_kws = {'fontsize':16, 'fontweight':'bold'})
sns.set(font_scale = 1.5)
plt.show()

# %%
## Show the most corrlated features with the output
uLimit =  0.1   # Upper threshold
lLimit = -0.1    # Lower threshold

mostCorrFeature = corrMatrix['Strength'][(corrMatrix['Strength'] > uLimit) | (corrMatrix['Strength'] < lLimit)]  ## Find the top correlated features (negative or positive)
mostCorrFeature = mostCorrFeature[:-1]                                                                           ## Delete the last row (Strength which is self-correlated)
print(mostCorrFeature)

# %%
## Drop the least correlated features 
corrFeatures = ['Cement', 'Blast Furnace Slag', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']  
corrDataFrame = dataFrame.drop(columns = ['Fly Ash'])
print('Data Frame Size :', corrDataFrame.shape)

# %% [markdown]
# ## Outliers 

# %%
def outliers(quartile, normal, Data, FeatureName, limit):
    global inx, med, upperLimit, lowerLimit

    ## Function to define outliers in the features based on the IQR
    if quartile == 1 and normal == 0:                       
        Q1 = Data[FeatureName].quantile(0.25)               # First quantile
        Q3 = Data[FeatureName].quantile(0.75)               # Third quantile
        IQR = Q3 - Q1
        lowerLimit = 0                                      # Lower limit should be zero
        upperLimit = Q3 + limit*IQR
    
        inx = np.where((Data[FeatureName] < lowerLimit) | (Data[FeatureName] > upperLimit))[0]
        med = Data[FeatureName].median()
        mea = Data[FeatureName].mean()
        
        print("Upper limit for", FeatureName , "feature = ", upperLimit)
        print("Lower limit for", FeatureName , "feature = ", lowerLimit)
        print("The Median for", FeatureName, "=", med)
        print("The Mean for", FeatureName, "=", mea)
        print('Number of outliers based on IQR for', FeatureName, 'is', len(inx))
    
    ## Function to define outliers in the features based on the Z-score
    elif quartile == 0 and normal == 1:                     
        Z_score = stats.zscore(Data[FeatureName])    
        inx = np.where(np.abs(Z_score) > limit)[0]
        med = Data[FeatureName].median()
        mea = Data[FeatureName].mean()
        
        upperLimit = Data[FeatureName].mean() + limit*Data[FeatureName].std()
        lowerLimit = Data[FeatureName].mean() - limit*Data[FeatureName].std()
        
        print("Upper limit for", FeatureName ,"feature = ", upperLimit)
        print("Lower limit for", FeatureName ,"feature = ", lowerLimit)
        print("The Median for", FeatureName, "=", med)
        print("The Mean for", FeatureName, "=", mea)
        print('Number of outliers based on Z-score for', FeatureName, 'is', len(inx))
    elif quartile == 0 and normal == 0:
        print('--------------No Action--------------')
        print('The matrix shape is', Data.shape)
    else:
        print('Wrong')

# %%
def decision(dropRow, capMedian, capMM, Data, FeatureName, inx, med, lowerLimit, upperLimit):

    if dropRow == 1 and capMedian == 0 and capMM == 0:     
    ## Dropping the outliers
        Data.drop(inx, inplace = True)
        Data.reset_index(drop = True, inplace = True)
        print('--------------The outliers will be dropped--------------')
        print('The matrix shape is', Data.shape)
    elif dropRow == 0 and capMedian == 1 and capMM == 0:
    ## Capping  the outliers with median
        Data[FeatureName] = np.where((Data[FeatureName] > upperLimit) | (Data[FeatureName] < lowerLimit), med, Data[FeatureName])
        print('--------------The outliers will be capped with mMdian--------------')
        print('The matrix shape is', Data.shape)
    elif dropRow == 0 and capMedian == 0 and capMM == 1:
        ## Capping  the outliers with max and min
        Data[FeatureName] = np.where(Data[FeatureName] > upperLimit, upperLimit, np.where(Data[FeatureName] < lowerLimit, lowerLimit, Data[FeatureName]))
        print('--------------The outliers will be capped with Min and Max--------------')
        print('The matrix shape is', Data.shape)
    elif dropRow == 0 and capMedian == 0 and capMM == 0:
        print('--------------No Action--------------')
        print('The matrix shape is', Data.shape)
    else:
        print('Wrong')
        
    return Data               ## Return the new modified matrix with no outliers

# %%
outliers(1, 0, corrDataFrame, "Cement", 1.5)
cleanDataFrame = decision(0, 0, 0, corrDataFrame, "Cement", inx, med, lowerLimit, upperLimit)

# %%
outliers(1, 0, cleanDataFrame, "Blast Furnace Slag", 1.5)
cleanDataFrame = decision(0, 0, 0, cleanDataFrame, "Blast Furnace Slag", inx, med, lowerLimit, upperLimit)

# %%
outliers(1, 0, cleanDataFrame, "Water", 1.5)
cleanDataFrame = decision(0, 0, 0, cleanDataFrame, "Water", inx, med, lowerLimit, upperLimit)

# %%
outliers(1, 0, cleanDataFrame, "Superplasticizer", 1.5)
cleanDataFrame = decision(1, 0, 0, cleanDataFrame, "Superplasticizer", inx, med, lowerLimit, upperLimit)

# %%
outliers(1, 0, cleanDataFrame, "Coarse Aggregate", 1.5)
cleanDataFrame = decision(0, 0, 0, cleanDataFrame, "Coarse Aggregate", inx, med, lowerLimit, upperLimit)

# %%
outliers(1, 0, cleanDataFrame, "Fine Aggregate", 1.5)
cleanDataFrame = decision(1, 0, 0, cleanDataFrame, "Fine Aggregate", inx, med, lowerLimit, upperLimit)

# %%
outliers(1, 0, cleanDataFrame, "Age", 1.5)
cleanDataFrame = decision(1, 0, 0, cleanDataFrame, "Age", inx, med, lowerLimit, upperLimit)

# %% [markdown]
# ## Dataframe statistics with outliers 

# %%
dataFrame.describe()

# %% [markdown]
# ## Dataframe statistics without outliers (Cleaned)

# %%
cleanDataFrame.describe()

# %%
## Data Standardization
features_wTarget = np.append(corrFeatures, 'Strength')          ## Add the target "Strength"

standardDataFrame = cleanDataFrame.copy()                       ## clean data frame includes data after outliers removal including the target.
standardDataFrame[features_wTarget] = StandardScaler().fit_transform(standardDataFrame[features_wTarget])

## Data Normalization
normalizeDataFrame = cleanDataFrame.copy()
normalizeDataFrame[features_wTarget] = MinMaxScaler().fit_transform(normalizeDataFrame[features_wTarget])

# %% [markdown]
# ## Standardized Data 

# %%
standardDataFrame.describe()

# %% [markdown]
# ## Normalized Data 

# %%
normalizeDataFrame.describe()

# %% [markdown]
# ## Machine Learning Model

# %%
## (Based on Standardization) Define the inputs, output, and split the data for training and testing
X = standardDataFrame.drop(columns = ['Strength'])                                                                                                # Define the input
Y = standardDataFrame.drop(columns = ['Cement', 'Blast Furnace Slag', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age','Water'])   # Define the output
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)                                                     # Split the data into test data and training data 

## Define the Linear Regression Model
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(x_test)

# %%
## (Based on Standardization) Calculating the RMS Error, Model Coefficient, and Interception
print('RMS :', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Intercept :', model.intercept_)
print('Accuracy :', r2_score(y_test, y_pred))
print('Coefficents :', model.coef_)
print('--------------------')

## Plot the predicted results vs. testing result
plt.figure(figsize = (8, 8))
plt.scatter(y_test, y_pred, s = 50, c = 'r', edgecolors = 'k')
plt.xlabel("Tested Value", fontsize = 20, family = 'serif')
plt.ylabel("Predicted Value", fontsize = 20, family = 'serif')
plt.show()

# %%
## Return the de-Standardization predictions

y_test_df = pd.DataFrame(y_test)
Y_train_df = pd.DataFrame(Y_train)
y_pred_df=pd.DataFrame(y_pred)
Y = cleanDataFrame.copy()                                   ## raw data after cleaning outliers and Nan if any 
Y = Y.drop(columns = ['Cement', 'Blast Furnace Slag', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age', 'Water'])
y_test_df_actual = y_test_df * (Y.std()) + Y.mean() ## Test Data without standardization 

y_std  = Y.std()
y_mean = Y.mean()
y_pred_df_actual= y_pred_df*(y_std[0]) +y_mean[0]   ## predicted value revert to without standardization

# %%
## (Based on Normalization) Define the inputs, output, and split the data for training and testing
XN = normalizeDataFrame.drop(columns = ['Strength'])                                                                                                 # Define the input
YN = normalizeDataFrame.drop(columns = ['Cement', 'Blast Furnace Slag', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age', 'Water'])   # Define the output

X_trainN, x_testN, Y_trainN, y_testN = train_test_split(XN, YN, test_size = 0.2, random_state = 42)                   # Split the data into test data and training data 

## Define the Linear Regression Model
modelN = linear_model.LinearRegression()          
modelN.fit(X_trainN, Y_trainN)
y_predN = modelN.predict(x_testN)

# %%
## (Based on Normalization) Calculating the RMS Error, Model Coefficient, and Interception 
print('RMS :', np.sqrt(mean_squared_error(y_testN, y_predN)))
print('Interception :', modelN.intercept_)
print('Accuracy :', r2_score(y_testN, y_predN))
print('Coefficients :', modelN.coef_)
print('--------------------')

## Plot the predicted results vs. testing result
plt.figure(figsize = (8, 8))
plt.scatter(y_testN, y_predN, s = 50, c = 'r', edgecolors = 'k')
plt.xlabel("Tested Value", fontsize = 20, family = 'serif')
plt.ylabel("Predicted Value", fontsize = 20, family = 'serif')
plt.show()

# %% [markdown]
# # PCA

# %% [markdown]
# ### Dimension Analysis

# %%
## Drop the output and standardize the complete feature dataframe 
PCADataFrame = dataFrame.copy()                 
featuresDataFrame = PCADataFrame.drop(columns = ['Strength'])
target = PCADataFrame.drop(columns = ['Cement', 'Blast Furnace Slag' , 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age', 'Water', 'Fly Ash'])

standardDataFramePCA_features = StandardScaler().fit_transform(featuresDataFrame)       ## Standardization
z = standardDataFramePCA_features.T
covMatrix = np.cov(z)                               ## Calulate the covariance matrix

print('The size of covariance matrix is :', covMatrix.shape)

## Feature analysis for PCA  
eigValue, eigVector = np.linalg.eig(covMatrix)      ## Calculate the eigenvalue and eigenvector for the covariance matrix
inx = np.argsort(eigValue)[::-1]                    ## The index of eigenvalues components in decending order
ordEigValues = eigValue[inx]                        ## Re-arrange eigenvalues
ordEigVector = eigVector[:,inx]                     ## Re-arrange eigenvector
featureWeight = ordEigValues/ordEigValues.sum()     ## Weight for each feature (variance ratio)
featureWeight

# %%
## Display the input dataframe for PCA
featuresDataFrame.describe()

# %%
print('The index order for eigenvalues is :', inx)  

# %%
## Plot the PCA dimension analysis

csum = np.cumsum(featureWeight)
xint = range(1, len(csum) + 1)
plt.figure(figsize = (7, 7))
plt.plot(xint,csum, marker = 'o', markersize = 10, linestyle = '-', color = 'k', linewidth = '2.5')

plt.xlabel("Number of features", fontsize = 20, family = 'serif')
plt.ylabel('Cumulative variance (%)', fontsize = 20, family = 'serif')
plt.show()

# %%
## Apply the Principal Component Analysis (PCA) (Dimension = 2)

pcaModel = PCA(n_components = 2)                                      # Define the PCA model 
pcaModelFitTransform = pcaModel.fit_transform(featuresDataFrame)       
print(pcaModel.explained_variance_ratio_)

# %%
print ('The PCA model dimension for inputs is :', pcaModelFitTransform.shape)
print ('The PCA model dimension for outputs is :', target.shape)

# %%
X_trainPCA, x_testPCA, Y_trainPCA, y_testPCA = train_test_split(pcaModelFitTransform, target, test_size = 0.2, random_state = 42)      # Split the data into test data and training data 

## Define the Linear Regression Model
modelPCA = linear_model.LinearRegression()          
modelPCA.fit(X_trainPCA, Y_trainPCA)
y_predPCA = modelPCA.predict(x_testPCA)

# %%
## (Based on Normalization) Calculating the RMS Error, Model Coefficient, and Interception 

print('RMS :', np.sqrt(mean_squared_error(y_testPCA, y_predPCA)))
print('Interception :', modelPCA.intercept_)
print('Accuracy :', r2_score(y_testPCA, y_predPCA))
print('Coefficients :', modelPCA.coef_)
print('--------------------')

# %%
## Plot the predicted results vs. testing result

plt.figure(figsize = (8, 8))
plt.scatter(y_testPCA, y_predPCA, s = 50, c = 'r', edgecolors = 'k')
plt.xlabel("Tested Value", fontsize = 20, family = 'serif')
plt.ylabel("Predicted Value", fontsize = 20, family = 'serif')
plt.show()

# %% [markdown]
# Reverse data from standardized to actual valeus of strength

# %%
## Return the de-standardization predictions

Y_train_df = pd.DataFrame(Y_train)                          ## Converting numpay into dataframe 
y_test_df = pd.DataFrame(y_test)                            ## Converting numpay into dataframe
y_pred_df = pd.DataFrame(y_pred)                            ## Converting numpay into dataframe
                       

Y = cleanDataFrame.copy()                                   ## Make a copy of the clean dataframe 
Y = Y.drop(columns = ['Cement', 'Blast Furnace Slag', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age', 'Water'])
y_std  = Y.std()                                            ## The standard deviation for the 'Strength' 
y_mean = Y.mean()                                           ## The mean for the 'Strength'

# Outputs -Linear Regression- 
y_test_df_actual = y_test_df * (Y.std()) + Y.mean()         ## Test data without standardization
y_pred_df_actual = y_pred_df * (y_std[0]) + y_mean[0]       ## Prediction data without standardization
y_pred_df_actual.columns =['Predicted-linear Regression']

## PCA output 
y_predPCA_df = pd.DataFrame(y_predPCA)

y_predPCA_df .columns =['Predicted-PCA']

# %% [markdown]
# printing predictions to CSV

# %%
y_pred_df_actual.to_csv('predictions_linearRegression.csv')
y_predPCA_df .to_csv('predictions_PCA.csv')



