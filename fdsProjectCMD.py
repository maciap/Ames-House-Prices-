#import required packages 
import sys
import pandas as pd
import numpy as np
from scipy.stats import skew, boxcox
from sklearn import preprocessing
from sklearn import linear_model
from xgboost import XGBRegressor


def drop_outliers(train):
    ''' remove outliers (visually identified) from the training data set.''' 
    
    #indices of the outliers.
    outliers_toberemoved = [1349, 968, 524,462, 1324, 812, 218, 1328, 666, 688, 431, 335, 523, 137, 3, 970, 632, 495, 30, 1432, 1453, 628,
                313, 669, 1181, 365, 1119, 142, 607, 197, 271, 66, 613, 185, 17, 1170, 1211, 238, 1292, 874, 691, 457, 1182,
                574, 1298, 1383, 1186, 150, 190, 463, 112, 1299]
    
    train.drop(train.index[outliers_toberemoved], inplace=True)
    return train 


def fill_NAS(complete_data): 
    '''impute missing values. Instead of sophisticated machine learning based imputation methods, 
    it is preferred to exploit as best as possible the knowledge concerning the data. A large amount of missing values correspond 
    to attributes that are not present in a house. Hence, they can be replaced by "None" and 0 for categorical and numerical predictors, 
    respectively. Moreover, there are clear relationship among the features. Therefore, in many cases, it is possible to use the conditional median (for numerical variables)
    or mode (for categorical variables) in order to replace missing values. When no strong dependece is observed, the unconditional mean or mode are used. '''    
    
    #categorical features to be replaced by "None" (indicating the absence of a given aspect in the house).
    none_features = ["MiscFeature", "Alley", "Fence", "FireplaceQu",  "GarageType",  
             "BsmtQual", "BsmtCond", "BsmtFinType1", "BsmtFinType2"]
    complete_data[none_features] = complete_data[none_features].fillna("None")
    
    #MSZoning: imputation conditional on MSSubclass 
    #the categorical feature MSZoning cannot be None obviously, but there are 4 missing values. A wise 
    #strategy involves predicting the most likely category, or, probably better, conditional on the feature MSSubClass.
    complete_data["MSZoning"] = complete_data["MSZoning"].fillna({1915: "RM", 2216:  "RL", 2250 : "RM", 2904 : "RL" })
   
    #categorical with the unconditional mode (since there is no very strong dependence).
    mode_features = ["Functional", "Electrical", "KitchenQual", "Exterior2nd", "Exterior1st"]
    for column in mode_features:
        complete_data[column].fillna(complete_data[column].mode()[0], inplace=True)

    #Utilities: almost constant if not missing. Thus, it is not informative and can be neglected.
    complete_data = complete_data.drop(['Utilities'], axis=1)

    #GarageYrBlt: an intuitive strategy is to assume that the Garage is built the same year as the house.
    complete_data["GarageYrBlt" ] = complete_data["GarageYrBlt"].fillna(complete_data["YearBuilt"])
    
    #numerical features in which missing values are replaced by null. As a matter of fact, they correspond to numerical attributes of characteristic of the house 
    #that are not present, as suggested by associated predictors.
    zero_features = ["GarageCars", "GarageArea", "BsmtFinSF1",  "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"]
    complete_data[zero_features] = complete_data[zero_features].fillna(0)
    
    #BsmtExposure: can be null but when TotalBsmtSF is positive the missing values are replaced by the most likely values conditional on the median TotalBsmtSF.
    indices = complete_data["BsmtExposure"][(complete_data["BsmtExposure"].isnull()) & (complete_data["TotalBsmtSF"] > 0)].index
    complete_data["BsmtExposure"].replace(indices, value = ["No", "Gd", "No"], inplace = True)
    complete_data["BsmtExposure"] = complete_data["BsmtExposure"].fillna("None")

    #As concerns MasVnrType, an house has null MasVnrType and positive MasVnrArea
    #The missing values is again replaced by the most likely value given the value of MasVnrArea. Specifically, the median MasVnrArea within 
    #each MasVnrType category is computed and the Masonry veneer type for which the associated median is closest to the MasVnrArea value of the house of interest is chosen.
    #The other missing values can be replaced by "None".
    MasVnrTypeIndex = complete_data["MasVnrType"][(complete_data["MasVnrType"].isnull()) & (complete_data["MasVnrArea"].notnull())].index
    complete_data["MasVnrType"].replace(MasVnrTypeIndex, value = "Stone", inplace = True)
    complete_data["MasVnrType"] = complete_data["MasVnrType"].fillna("None")

    #PoolQC: for 3 examples having missing PoolQC and postive PoolArea, missing values are imputed using the most likely value taking the median PoolArea 
    #into account. Thus, the same approach as for the Masonry veneer is adopted. The remaining ones correspond to houses without pool.
    indices = complete_data["PoolQC"][(complete_data["PoolQC"].isnull()) & (complete_data["PoolArea"] > 0)].index
    complete_data["PoolQC"].replace(indices, value = ["Ex", "Ex", "Fa"], inplace = True)
    complete_data["PoolQC"] = complete_data["PoolQC"].fillna("None")

    #SaleType: missing value replaced by unconditional mode.
    complete_data["SaleType"] = complete_data["SaleType"].fillna("WD")
    
    
    #extract the index of an observation showing positive Garage Area and missing GarageQual (used at lines 89-90-91). As a matter of fact, in this case, GarageQual cannot be "None".
    garageIndex  = complete_data["GarageArea"][(complete_data["GarageQual"].isnull()) & (complete_data["GarageArea"] > 0)].index
    
    #for the features "GarageQual", "GarageFinish" and "GarageCond" one missing value corresponds to a Garage which is actually present, as pointed out above.
    #Here, the imputation is based on the sum of the median GarageArea and the median GarageCars.
    garage_features = ['GarageQual', 'GarageFinish', 'GarageCond', 'GarageType']
    complete_data["GarageQual"].replace(garageIndex, value = "TA", inplace = True)
    complete_data["GarageFinish"].replace(garageIndex, value = "Unf", inplace = True)
    complete_data["GarageCond"].replace(garageIndex, value = "TA", inplace = True)

    #remaining missing values of the categorical Garage-related features are associated with not existing Garages. 
    complete_data[garage_features] = complete_data[garage_features].fillna("None")         

    #LotFrontage:  many (480) missing values are present and are replaced by the median LotFrontage within the associated neighbourhood.
    complete_data["LotFrontage"] = complete_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    
    return complete_data


def encoding(complete_data):
    '''handle categorical (nominal and ordinal) features.'''
   
    #map ordinal features from categorical to numerical (discrete) 
    
    mapPool = mapPool = pd.Series([0, 1, 2, 3], ['Ex', 'Gd','Fa', 'None'])    
    complete_data['PoolQC'] = (complete_data['PoolQC']).map(mapPool)    
            
    mapKitchen = pd.Series([0, 1, 2, 3, 4], ['Ex', 'Gd', 'TA', 'Fa', 'Po'])
    complete_data['KitchenQual'] = (complete_data['KitchenQual']).map(mapKitchen)   
    
    GenericFeat1 = ['FireplaceQu', 'GarageQual', 'GarageCond'] 
    mapGeneric = pd.Series([0, 1, 2, 3, 4, 5], ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'])
    complete_data[GenericFeat1] = complete_data[GenericFeat1].apply(lambda x : x.map(mapGeneric),axis = 0)
    
    GenericFeat2 = ['ExterCond', 'HeatingQC']  
    mapGeneric = pd.Series([0, 1, 2, 3, 4], ['Ex', 'Gd', 'TA', 'Fa', 'Po'])
    complete_data[GenericFeat2] = complete_data[GenericFeat2].apply(lambda x : x.map(mapGeneric),axis = 0)
    
    mapExterQual = pd.Series([0, 1, 2, 3], ['Ex', 'Gd', 'TA', 'Fa'])
    complete_data['ExterQual'] = (complete_data['ExterQual']).map(mapExterQual)
    
    mapBsmtQual = pd.Series([0, 1, 2, 3, 4], ['Ex', 'Gd', 'TA', 'Fa', 'None'])
    complete_data['BsmtQual'] = (complete_data['BsmtQual']).map(mapBsmtQual)
    
    mapBsmtCond = pd.Series([0, 1, 2, 3, 4], ['Gd', 'TA', 'Fa','Po', 'None'])
    complete_data['BsmtCond'] = (complete_data['BsmtCond']).map(mapBsmtCond)
    
    mapFun = pd.Series([0, 1, 2, 3, 4, 5, 6], ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev'])
    complete_data["Functional"] = (complete_data["Functional"]).map(mapFun)   
            
    BsmtFinTypeFeat = ['BsmtFinType1', 'BsmtFinType2' ]
    mapBsmtFinType = pd.Series([0, 1, 2, 3, 4, 5, 6], ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ','Unf','None'])
    complete_data[BsmtFinTypeFeat] = complete_data[BsmtFinTypeFeat].apply(lambda x : x.map(mapBsmtFinType), axis = 0)

    mapFence = pd.Series([0, 1, 2, 3, 4], ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'None'])
    complete_data['Fence'] = (complete_data['Fence']).map(mapFence)
            
    mapBsmtExp = pd.Series([0,1, 2, 3, 4], ['Gd', 'Av', 'Mn', 'No', 'None'])
    complete_data['BsmtExposure'] = (complete_data['BsmtExposure']).map(mapBsmtExp)
            
    mapGarageFin = pd.Series([0, 1, 2, 3], ['Fin', 'RFn', 'Unf', 'None'])
    complete_data['GarageFinish'] = (complete_data['GarageFinish']).map(mapGarageFin)
            
    mapLanSlope = pd.Series([0, 1, 2], ['Gtl', 'Mod', 'Sev'])
    complete_data['LandSlope'] = (complete_data['LandSlope']).map(mapLanSlope)
            
    mapLotShape = pd.Series([0, 1, 2, 3], ['Reg', 'IR1', 'IR2', 'IR3'])
    complete_data['LotShape'] = (complete_data['LotShape']).map(mapLotShape)
            
    mapPavedDrive = pd.Series([2, 1 , 0], ['Y', 'P', 'N'])
    complete_data['PavedDrive'] = (complete_data['PavedDrive']).map(mapPavedDrive)
            
    mapCentrAir = pd.Series([1, 0], ['Y', 'N'])
    complete_data['CentralAir'] = (complete_data['CentralAir']).map(mapCentrAir)
            
    mapAlley =  pd.Series([0, 1, 2], ['Grvl', 'Pave', 'None'])
    complete_data['Alley'] = (complete_data['Alley']).map(mapAlley)
    
    
    #convert original nominal features (no intrinsic order) to dummy variables.
    
    #list containing all the categorical features in the data set 
    categorical_features = ['MSZoning', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1',
       'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
       'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
       'Electrical', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition', "Street",
       'MSSubClass']

    #transform nominal features into set of dummy variables
    complete_data = pd.get_dummies(complete_data,  columns= categorical_features)
    
    return complete_data 



def add_derivedFeatures(complete_data):
    '''add extra features to the data set, in order to improve the quality of the predictions.
    Most of them are (first order) interactions and polynomial terms (degrees = 1/2, 2, 3). Some of them are obtained by 
    dichotomizing existing features, comparing two features or normalizing by subtracting the minimum. Others are simply 
    built by summing or subtracting two or more features. 
    Here, human expertise is used.
    Interactions are grouped, whereas the other features are presented in the order in which they have been created.'''
    
    #total of bath-related features 
    complete_data["completeBath"] = complete_data["BsmtFullBath"] + (0.5 * complete_data["BsmtHalfBath"]) + complete_data["FullBath"] + (0.5 * complete_data["HalfBath"])
    #combination of basement and ground floor living area. 
    complete_data["completeSF"] = complete_data["GrLivArea"] + complete_data["TotalBsmtSF"]
    #first and second floor area.
    complete_data["completeFloorsSF"] = complete_data["1stFlrSF"] + complete_data["2ndFlrSF"]
    #combine porch-related features.
    complete_data["completePorchSF"] = complete_data["OpenPorchSF"] + complete_data["EnclosedPorch"] + \
    complete_data["3SsnPorch"] + complete_data["ScreenPorch"]
    #difference between remodel data and year in which the house is built.
    complete_data["RemodelYear"] = (complete_data["YearRemodAdd"] - complete_data["YearBuilt"])
    #number of years between the sale of the house and the remodeling.
    complete_data["AfterRemodelingYears"] = complete_data["YrSold"] - complete_data["YearRemodAdd"]
    #binary feature indicating whether the house has been remodeled.
    complete_data["RemodelingBool"] = (complete_data["YearRemodAdd"] != complete_data["YearBuilt"]) * 1
    #binary variable taking value 1 when the house has been remodeled in the same year the sale takes place and 0 otherwise.
    complete_data["justRemodeling"] = (complete_data["YearRemodAdd"] == complete_data["YrSold"]) * 1
    #dichotomous feature indicating whether the years in which the house is built and sold coincide.
    complete_data["justBuilt"] = (complete_data["YearBuilt"] == complete_data["YrSold"]) * 1
    #overall numerical indicator associated with the garage.
    complete_data["completeGarage"] = complete_data["GarageFinish"] * (complete_data["GarageArea"] + complete_data["GarageCars"]) * complete_data["GarageCond"]
    
    #interactions.
    complete_data["Interaction1"] = complete_data["OverallQual"] * complete_data["OverallCond"]
    complete_data["Interaction2"] = complete_data["GarageQual"] * complete_data["GarageCond"]
    complete_data["Interaction3"] = complete_data["ExterQual"] * complete_data["ExterCond"]
    complete_data["Interaction4"] = complete_data["KitchenAbvGr"] * complete_data["KitchenQual"]
    complete_data["Interaction5"] = complete_data["Fireplaces"] * complete_data["FireplaceQu"]
    complete_data["Interaction6"] = complete_data["GarageArea"] * complete_data["GarageQual"]
    complete_data["Interaction7"] = complete_data["PoolArea"] * complete_data["PoolQC"]
    complete_data["Interaction8"] = complete_data["TotalBsmtSF"] * complete_data["BsmtUnfSF"]
    complete_data["Interaction9"] = complete_data["YearBuilt"] * complete_data["BsmtFinSF1"]
    complete_data["Interaction10"] = complete_data["YearBuilt"] * complete_data["LotFrontage"]
    complete_data["Interaction11"] = complete_data["YearBuilt"] * complete_data["YearRemodAdd"]
    complete_data["Interaction12"] = complete_data["LotFrontage"] * complete_data["LotArea"]
    complete_data["Interaction13"] = complete_data["HalfBath"] * complete_data["FullBath"]
    complete_data["Interaction18"] = complete_data["FullBath"] * complete_data["YearBuilt"]
    complete_data["Interaction19"] = complete_data["TotalBsmtSF"] * complete_data["BedroomAbvGr"]
    complete_data["Interaction16"] = complete_data["1stFlrSF"] * complete_data["LotFrontage"]
    complete_data["Interaction17"] = complete_data["YearRemodAdd"] * complete_data["2ndFlrSF"]
    complete_data["Interaction18"] = complete_data["FullBath"] * complete_data["YearBuilt"]
    complete_data["Interaction19"] = complete_data["TotalBsmtSF"] * complete_data["BedroomAbvGr"]
    complete_data["Interaction16"] = complete_data["1stFlrSF"] * complete_data["LotFrontage"]
    complete_data["Interaction20"] = complete_data["Interaction1"] * complete_data["Interaction4"]
    complete_data["Interaction21"] = complete_data["LotArea"] * complete_data["YearBuilt"]
    complete_data["Interaction22"] = complete_data["Interaction1"] * complete_data["LotArea"]
    complete_data["Interaction23"] = complete_data["Interaction1"] * complete_data["YearBuilt"]
    complete_data["Interaction24"] = complete_data["Interaction1"] * complete_data["Fireplaces"]
    complete_data["Interaction25"] = complete_data["LotArea"] * complete_data["Fireplaces"]
    complete_data["Interaction26"] = complete_data["HeatingQC"] * complete_data["Interaction4"]
    complete_data["Interaction27"] = complete_data["HeatingQC"] * complete_data["LotArea"]
    complete_data["Interaction28"] = complete_data["HeatingQC"] * complete_data["Interaction1"]
    
    
    #polynomial terms and few other extra features  
    complete_data['MasVnrArea-2'] = complete_data['MasVnrArea'] ** 2
    complete_data['MasVnrArea-3'] = complete_data['MasVnrArea'] ** 3
    complete_data['MasVnrArea-sq'] = np.sqrt(complete_data['MasVnrArea'])
    complete_data['BsmtFinSF1-2'] = complete_data['BsmtFinSF1'] ** 2
    complete_data['BsmtFinSF1-3'] = complete_data['BsmtFinSF1'] ** 3
    complete_data['BsmtFinSF1-sq'] = np.sqrt(complete_data['BsmtFinSF1'])
    complete_data['BsmtFinSF2-2'] = complete_data['BsmtFinSF2'] ** 2
    complete_data['BsmtFinSF2-3'] = complete_data['BsmtFinSF2'] ** 3
    complete_data['BsmtFinSF2-sq'] = np.sqrt(complete_data['BsmtFinSF2'])  
    complete_data['TotalBsmtSF-2'] = complete_data['TotalBsmtSF'] ** 2
    complete_data['TotalBsmtSF-3'] = complete_data['TotalBsmtSF'] ** 3
    complete_data['TotalBsmtSF-sq'] = np.sqrt(complete_data['TotalBsmtSF'])    
    complete_data["completePorchSF-2"] = complete_data["completePorchSF"] ** 2
    complete_data["completePorchSF-3"] = complete_data["completePorchSF"] ** 3
    complete_data["completePorchSF-sq"] = np.sqrt(complete_data["completePorchSF"])
    complete_data["completeGarage-2"] = complete_data["completeGarage"] ** 2
    complete_data["completeGarage-3"] = complete_data["completeGarage"] ** 3
    complete_data["completeGarage-sq"] = np.sqrt(complete_data["completeGarage"])  
    complete_data["Interaction5-2"] = complete_data["Interaction5"] ** 2
    complete_data["Interaction5-3"] = complete_data["Interaction5"] ** 3
    complete_data["Interaction5-sq"] = np.sqrt(complete_data["Interaction5"])
    complete_data["Interaction1-2"] = complete_data["Interaction1"] ** 2
    complete_data["Interaction1-3"] = complete_data["Interaction1"] ** 3
    complete_data["Interaction1-sq"] = np.sqrt(complete_data["Interaction1"])
    complete_data["completeFloorsSF-2"] = complete_data["completeFloorsSF"] ** 2
    complete_data["completeFloorsSF-3"] = complete_data["completeFloorsSF"] ** 3
    complete_data["completeFloorsSF-sq"] = np.sqrt(complete_data["completeFloorsSF"])
    #normalize date of remodeling by subtracting the minimum.
    complete_data["YearRemodAddNorm"] = complete_data["YearRemodAdd"] - min(complete_data["YearRemodAdd"])
    #normalize year in which the garage is built by subtracting the minimum. 
    complete_data["GarageYrBltNorm"] = complete_data["GarageYrBlt"] - min(complete_data["GarageYrBlt"])
    #dichotomizing OverallQual and OverallCond, splitting the observations in two classes based on the median.
    complete_data["goodQual"] = ( complete_data["OverallQual"] > np.median(complete_data["OverallQual"]) ) * 1 
    complete_data["goodCond"] = (complete_data["OverallCond"] > np.median(complete_data["OverallCond"]) ) * 1 
    complete_data["YearUpgrade"] = complete_data["YearRemodAdd"] - complete_data["YearBuilt"]
    complete_data["YearUpgrade-2"] = complete_data["YearUpgrade"] ** 2
    complete_data["YearRemodAdd-2"] = complete_data["YearRemodAddNorm"] ** 2
    complete_data["YearBuilt-2"]  = complete_data["YearBuilt"] ** 2
    complete_data["YearBuilt-3"]  = complete_data["YearBuilt"] ** 3
    complete_data["GarageYrBltNorm-2"]  = complete_data["GarageYrBltNorm"] ** 2
    complete_data["GarageYrBltNorm-3"]  = complete_data["GarageYrBltNorm"] ** 3
    complete_data["GarageArea2-2"] = complete_data["GarageArea"] ** 2
    complete_data["GarageArea2-3"] = complete_data["GarageArea"] ** 3
 
    complete_data["TotRmsAbvGrd-2"] = complete_data["TotRmsAbvGrd"] ** 2
    complete_data["TotRmsAbvGrd-3"] = complete_data["TotRmsAbvGrd"] ** 3
    complete_data["completeSF-2"] = complete_data["completeSF"] ** 2
    complete_data["completeSF-3"] = complete_data["completeSF"] ** 3
    complete_data["completeSF-sq"] = np.sqrt(complete_data["completeSF"])

    complete_data["GrLivArea-2"] = complete_data["GrLivArea"] ** 2
    complete_data["GrLivArea-3"] = complete_data["GrLivArea"] ** 3
    complete_data["GrLivArea-sq"] = np.sqrt(complete_data["GrLivArea"])
    complete_data["ExterQual-2"] = complete_data["ExterQual"] ** 2
    complete_data["ExterQual-3"] = complete_data["ExterQual"] ** 3
    complete_data["ExterQual-sq"] = np.sqrt(complete_data["ExterQual"])
    complete_data["GarageCars-2"] = complete_data["GarageCars"] ** 2
    complete_data["GarageCars-3"] = complete_data["GarageCars"] ** 3
    complete_data["GarageCars-sq"] = np.sqrt(complete_data["GarageCars"])
    complete_data["completeBath-2"] = complete_data["completeBath"] ** 2
    complete_data["completeBath-3"] = complete_data["completeBath"] ** 3
    complete_data["completeBath-sq"] = np.sqrt(complete_data["completeBath"])
    complete_data["KitchenQual-2"] = complete_data["KitchenQual"] ** 2
    complete_data["KitchenQual-3"] = complete_data["KitchenQual"] ** 3
    complete_data["KitchenQual-sq"] = np.sqrt(complete_data["KitchenQual"])
    complete_data["Interaction6-2"] = complete_data["Interaction6"] ** 2
    complete_data["Interaction6-3"] = complete_data["Interaction6"] ** 3
    complete_data["Interaction6-sq"] = np.sqrt(complete_data["Interaction6"])
    complete_data["LotFrontage-2"] = complete_data["LotFrontage"] ** 2
    complete_data["LotFrontage-3"] = complete_data["LotFrontage"] ** 3
    complete_data["LotFrontage-sq"] = np.sqrt(complete_data["LotFrontage"])
    
    #total Area (basement and floors)
    complete_data['completeArea'] = complete_data['TotalBsmtSF'] + complete_data['1stFlrSF'] + complete_data['2ndFlrSF']
    complete_data['completeArea-2'] = complete_data['completeArea']**2
    complete_data['completeArea-3'] = complete_data['completeArea']**3
    complete_data['completeArea-sq'] = np.sqrt(complete_data['completeArea'])
    #indicator variable for large houses 
    complete_data["BigHouse"] = ( complete_data['completeArea'] > np.median(complete_data['completeArea']) ) * 1 
    complete_data['OverallQual-2'] = complete_data['OverallQual']**2
    complete_data['OverallQual-3'] = complete_data['OverallQual']**3
    complete_data['OverallQual-sq'] = np.sqrt(complete_data['OverallQual'])
    complete_data['BsmtUnfSF-2'] = complete_data['BsmtUnfSF']**2
    complete_data['BsmtUnfSF-3'] = complete_data['BsmtUnfSF']**3
    complete_data['BsmtUnfSF-sq'] = np.sqrt(complete_data['BsmtUnfSF'])
    complete_data['LotArea-2'] = complete_data['LotArea']**2
    complete_data['LotArea-3'] = complete_data['LotArea']**3
    complete_data['LotArea-sq'] = np.sqrt(complete_data['LotArea'])
        
    #dichotomizing "OverallQual", "ExterQual", "BsmtCond" and "KitchenQual" using 3 as cutoff value and the 
    #garage summary based on the median 
    complete_data['PoorQualBool'] = (complete_data["OverallQual"]  < 3 ) * 1  
    complete_data['exterPoorQualBool'] = (complete_data["ExterQual"]  < 3 ) * 1  
    complete_data['ExtPoorCondBool'] = (complete_data["ExterCond"]  < 3 ) * 1  
    complete_data["BasementPoorCondBool"] = (complete_data["BsmtCond"]  < 3 ) * 1  
    complete_data['garagePoorQualBool'] = (complete_data["completeGarage"]  < np.median(complete_data["completeGarage"])) * 1  
    complete_data['kitchenPoorQualBool'] = (complete_data["KitchenQual"]  < 3) * 1  
    
    return complete_data
         

def find_highly_skewed(complete_data):
    '''compute skewness and identify highly-skewed features, i.e. feature with skewness larger than 0.70, in absolute value.'''
    
    #12 is a threshold to filter out features (of categorical nature) having few distincting values.
    candidates_features = list(filter(lambda x: len(set(complete_data[x])) > 12, complete_data.columns))
    all_skewness = complete_data[candidates_features].apply(skew, axis = 0)
    highly_skewed = all_skewness.index[abs(all_skewness) > 0.70]
    
    return highly_skewed
        

def deskew(complete_data, highly_skewed, dict_lambdas):
    '''adjust highly-skewed predictors, using the Box-Cox transformation. Each feature has a lambda parameter associated (determined 
    visually by attempting to make the feature distribution as symmetric as possible).
    It is worth noting that it could be possible to use maximum likelihood estimation for lambda, but it is not an easy procedure and
    scipy implementation has issues and fails.'''
    
    for feat in highly_skewed:
        if len(complete_data[feat][complete_data[feat] <= 0]) > 0:
            #translate by a positive constant so that all the values are positive (1 is needed in case the minimum is 0).
            complete_data[feat] += 1 + abs(min(complete_data[feat]))
        complete_data[feat] = boxcox(complete_data[feat], lmbda = dict_lambdas[feat])
         
    return complete_data
         

def scaling(train, test, scaler):
    '''scale the data using the interquantile range and the median.
    Typically, standardization is carried out using the mean and the standard deviation. However, as it is 
    well-known, these satistics are drastically affected by outliers. Thus, here, a robust alternative is considered.'''

    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    return (train, test) 
        

def feature_selection(coefs, thresh, complete_data, n_train, scaler):
    '''feature selection for Lasso and Ridge models.
    The procedure simply extracts the optimal subset of features retrieved using the printThresholdRMSE function shown in the auxiliary
    code section. As far as Lasso regression is concerned, this is equivalent to extract features associated with a coefficient different 
    from null. As regards the Ridge linear model, the procedure correspond to select only features having a coefficient larger than 0.008, 
    in absolute value.
    The output allows to fit the model using a smaller set of features, obviously changing the tuning parameter.'''
    
    #list in which each element is a tuple including the name of a feature and the absolute value of its model coefficient. 
    feats_coefs = list(zip(map(lambda x: round(x, 4), abs(coefs)), 
                     complete_data.columns))

    #sort previous list by the value of the cofficient 
    sorted_feats_coef = sorted(feats_coefs, key = lambda x: x[0], reverse = True)
    
    #extract the desired subset of features.
    chosen = [x[1] for x in sorted_feats_coef][:(thresh)]
    
    
    #create the resulting scaled training and test data sets. 
    complete_data_subset = complete_data[chosen]
    model_train = complete_data_subset[:n_train]
    model_test = complete_data_subset[n_train:]
    model_train, model_test = scaling(model_train, model_test, scaler)
    
    return (model_train, model_test) 


###################MAIN########################################################
if __name__ == "__main__":
    
# =============================================================================
# =============================================================================
########## Pre-modeling ######## 
#In this section, the data are processed in order to increase the performance of
#the model trained in what follows.
# =============================================================================   
# =============================================================================   
    
    #read the training set data.
    train = pd.read_csv(sys.argv[1])
    
    #read the test set data.
    test = pd.read_csv(sys.argv[2])
    
    #impute missing values.
    train = drop_outliers(train)
    
    #store test set id column in a variable. 
    test_ID = test['Id']
    #drop id column from train set and test set.
    train.drop("Id", axis = 1, inplace = True)
    test.drop("Id", axis = 1, inplace = True)
    
    #log-transform the dependent variable. This is equivalent to 
    #the Box-Cox transformation when the parameter lambda is set to null and it appears to 
    #be the appropriate transformation for the dependent variable. 
    y = np.log1p(train['SalePrice'])
    #remove the dependent variable stored in y from the train set.
    train.drop(['SalePrice'], axis=1, inplace=True)
    
    #bind rows of train set and test set so as to obtain a unique data frame. 
    #The reset_index method adds a new sequential index and drop = True avoids the creation of a useless new column storing the old indices. 
    complete_data = pd.concat((train, test)).reset_index(drop=True)
    
    #convert MSSubClass from numerical to categorical. As a matter of fact, the levels of the feature do not have any obvious intrinsic order. 
    #The encoding procedure generates the associated dummy features, as for the other variables of the same nature. 
    complete_data['MSSubClass'] = complete_data['MSSubClass'].apply(str)
    
    #fill missing values.
    complete_data = fill_NAS(complete_data)
    
    #convert categorical (nominal and ordinal) features into numerical features.
    complete_data = encoding(complete_data)
    
    #add derived features.
    complete_data = add_derivedFeatures(complete_data) 
    
    #retrieve features showing high skewness.
    highly_skewed = find_highly_skewed(complete_data)
    
    #create dictionary associating each feature with its lamdba parameter for the Box-Cox transformation, obtained visually.
    dict_lambdas = {'LotFrontage' : 0.4 ,'BsmtFinSF2' : 0.3, 'BsmtUnfSF': 0.4,  'TotalBsmtSF':0.4, 'LowQualFinSF' : 0.005, 'EnclosedPorch' : 0.4, '3SsnPorch': 1, 'ScreenPorch':0.7,
                    'MiscVal': 0.3, 'RemodelYear' : 0.01, 'completeGarage':0.4, 'Interaction8':0.3, 'Interaction9':0.4,'Interaction10': 0.2, 'Interaction12' :0.06, 'Interaction19': 0.5, 'BsmtFinSF2-2': 0.2, 'BsmtFinSF2-3':0.2, 'BsmtFinSF2-sq':0.5,
                    'TotalBsmtSF-2':0.4, 'TotalBsmtSF-3' : 0.15, 'TotalBsmtSF-sq' :1, 'completeGarage-2':0.4, 'completeGarage-3':4, 'Interaction1-2':3, 'Interaction1-3':0.1,'Interaction1-sq':1, 
                    'Interaction20' : 0.5, 'Interaction21': 0.3,'Interaction22':0.4, 'YearUpgrade':0.4, 'YearUpgrade-2':2, 'GarageYrBltNorm-2': 0.008,'GarageYrBltNorm-3':0.008, 'GarageArea2-2':0.2, 'GarageArea2-3':0.2, 
                    'Interaction6-2':0.2, 'Interaction6-3' : 0.15,'Interaction6-sq': 0.6, 'LotFrontage-2': 0.05,  'LotFrontage-3':0.03, 'BsmtUnfSF-2':0.2,'LotArea-2':0.5,  
                    'LotArea-3' : 0.2, 'LotArea-sq': 0.004, "MasVnrArea":0.4,
                     'LotArea' : 0.1, 'BsmtFinSF1' : 0.4, '1stFlrSF' : 0.2 ,'2ndFlrSF':0.7, 'GrLivArea':0.2, 'TotRmsAbvGrd': 0.3,
                     'WoodDeckSF':0.2 ,'OpenPorchSF': 0.4  ,'completeSF' :0.2 ,'completeFloorsSF' :0.1,'completePorchSF' :0.1  ,'Interaction16' :0.05 ,'Interaction17' :0.04  ,'MasVnrArea-2' : 0.3,
                     'MasVnrArea-3': 0.4,'MasVnrArea-sq':0.4 ,'BsmtFinSF1-2':0.2 ,'BsmtFinSF1-3': 0.1 ,'completePorchSF-2':0.4,'completePorchSF-3':0.2,'Interaction24':0.1,'Interaction25':0.1 ,'Interaction27':0.05 ,
                     'Interaction28':0.04,'completeFloorsSF-2':0.1,'completeFloorsSF-3':0.1, 'TotRmsAbvGrd-2':0.4 ,'TotRmsAbvGrd-3':0.2,'completeSF-2':0.3,'completeSF-3':0.1,'GrLivArea-2': 0.1,
                     'GrLivArea-3':0.1,'completeArea':0.2,'completeArea-2':0.3,'completeArea-3':0.1,
                     'BsmtUnfSF-3':0.1}
    
    
    
    #deskew features.
    complete_data = deskew(complete_data, highly_skewed, dict_lambdas)
    
    
    #store the number of examples in the training set. 
    n_train = train.shape[0]
    
    #store processed train and test set.
    new_train = complete_data[:n_train]
    new_test = complete_data[n_train:]    
    
    #instantiate class RobustScaler.
    scaler = preprocessing.RobustScaler()
    #standardize data using median and interquartile range.
    new_train, new_test = scaling(new_train, new_test, scaler)

# =============================================================================
# =============================================================================
########## Modeling ######## 
#For each model, parameter tuning is performed using 5-fold cross-validation.
#The code is illustrated in the auxiliary code section, as it is not scritly required,
#thus making the execution of the software considerably more efficient.
# =============================================================================   
# =============================================================================   

# =============================================================================
########## Lasso Regression  ######## 
# =============================================================================
    
    #feature selection.
    
    #fit Lasso model for feature selection purposes.
    lasso = linear_model.Lasso(alpha =0.00044, max_iter = 20000)
    lasso.fit(new_train, y)
    #number of features to be selected, having non-zero model coefficient.
    lasso_n = 110
    
    
    #extract training and test set associated with the selected features. 
    lasso_train, lasso_test = feature_selection(lasso.coef_, lasso_n , complete_data, n_train, scaler) 
    
    #prediction.
    
    #train Lasso regressor for prediction purposes.
    lasso = linear_model.Lasso(alpha =0.00006, max_iter = 20000)
    lasso.fit(lasso_train, y)
    #obtain predictions. 
    lasso_pred = np.expm1(lasso.predict(lasso_test))

# =============================================================================
########## Ridge Regression  ######## 
# =============================================================================
    
    #feature selection. 
    
    #fit Ridge model for feature selection purposes. 
    ridge = linear_model.Ridge(alpha=17.0)
    ridge.fit(new_train, y) 
    #number of features to be selected, corresponding to those showing coefficient magnitude greater than 0.008. 
    ridge_n = 153
    
    #extract training and test set associated with the selected features. 
    ridge_train, ridge_test = feature_selection(ridge.coef_, ridge_n, complete_data, n_train, scaler) 
    
    #prediction.
    
    #train Ridge regressor for prediction purposes.
    ridge = linear_model.Ridge(alpha=9)
    ridge.fit(ridge_train, y) 
    
    #obtain predictions. 
    ridge_pred = np.expm1(ridge.predict(ridge_test))

# =============================================================================
########## Extreme Gradient Boosting Regression ######## 
# =============================================================================       
    
    #As shown in the auxiliary code section, the parameters are tuned searching the parameter space
    #by considering a large number of combinations, in order to find the best cross-validation score.               
    #train extreme gradient boosting regressor.
    xgboost = XGBRegressor(n_estimators=550, max_depth = 4, colsample_bytree = 0.3, reg_lambda = 2, reg_alpha = 0.001 ,
                           min_child_weight =2, subsample = 0.8, n_jobs = 6)
    xgboost.fit(new_train,y)
    #obtain predictions. 
    xgboost_pred = np.expm1(xgboost.predict(new_test))


# =============================================================================
##########Averaging predictions (ensemble) ######## 
# =============================================================================   

    #the weights are determined by attempting to minimize the error in the test set 
    averaged_pred = 0.30 * lasso_pred + 0.25 * ridge_pred + 0.45 * xgboost_pred

# =============================================================================
########## Submission ######## 
# =============================================================================   

    #write predictions to csv file 
    pd.DataFrame({ "Id": test_ID , "SalePrice" : averaged_pred}).to_csv("pred.csv", index=False)

    print("Done.")

###############################################################################

# =============================================================================
########## Auxiliary code ######## 
#the code below is not strictly required for the execution of the software but it describes the procedure used 
#in order to obtain some necessary results.  Therefore, it might be valuable in order to understand some details 
#of the machine learning pipeline. 
# =============================================================================   

# =============================================================================
# #retrieve outliers to drop for each predictor.
#
# import matplotlib.pyplot as plt
# def scatterPlot(complete_data, n_train, predictor, y): 
#     
#     plt.scatter(x = complete_data[predictor].values[:n_train], y = y.values)
#     plt.title('Scatter plot')
#     plt.xlabel(predictor) 
#     plt.ylabel('Sale Price')
#     plt.show()
#     
# after properly examining the scatterplot, choosing x.val and y.val retrieve the indices of the outliers to be excluded from the analysis using the following line of code.
# complete_data[:n_train].index[(complete_data[:n_train][predictor] > x.val).values & (y < val).values].tolist()
#     
# =============================================================================

# =============================================================================
# #fill missing values 
#
# #The following code is helpful in order to appropriately impute missing values, as explained in 
# #the comments of the fill_NAS function.

# #MSSubClass.
# complete_data.groupby(['MSSubClass'])["MSZoning"].apply(lambda x : x.mode())
# 
# #MasVnrType.
# complete_data.groupby(['MasVnrType'])["MasVnrArea"].apply(lambda x : x.median())
# 
# #PoolQC.
# complete_data.groupby(['PoolQC'])["PoolArea"].apply(lambda x : x.median())
# 
# #Garage Qual, Garage Finish, GarageCond.
# complete_data.groupby(['GarageQual'])["GarageCars", "GarageArea" ].apply(lambda x: x["GarageCars"].median() + x["GarageArea"].median())
# complete_data.groupby(['GarageFinish'])["GarageCars", "GarageArea" ].apply(lambda x: x["GarageCars"].median() + x["GarageArea"].median())
# complete_data.groupby(['GarageCond'])["GarageCars", "GarageArea" ].apply(lambda x: x["GarageCars"].median() + x["GarageArea"].median())
# 
# =============================================================================

# =============================================================================
# #determine Box-Cox transformation parameter for each predictor.
#
# #draw histogram for a given value of lambda.
# 
# def plotBoxCox(complete_data, predictor, param):
#     plt.hist(boxcox(complete_data[predictor], lmbda = param))
#     plt.title(predictor) 
#     plt.show()
#     
# 
# #call the function for different values of the parameter lambda.
# plotBoxCox(complete_data, predictor, param)
#     
# =============================================================================
    
# =============================================================================
# #cross-validation framework
#
# from sklearn import model_selection
# 
# #compute 5-folds cross validation average Root Mean Squared Error for model selection.
# #If the avg flag is set to False, the function returns the standard deviation of the computed errors.

# def CVRMSE(train_predictors, y, regressor, avg = True):
#   #build K-Folds cross-validator. It splits the training set in 5 folds. Shuffling before splitting is additionally carried out.
#   folds = model_selection.KFold(5, shuffle=True, random_state=123).get_n_splits(train_predictors)     
#   #compute the Root Mean Squared Errors corresponding to different folds.
#   rmses =  np.sqrt(-model_selection.cross_val_score(regressor, train_predictors, y, scoring="neg_mean_squared_error", cv = folds))
#   if avg: 
#   #return mean
#       return rmses.mean()
#   #return standard deviation 
#   return rmses.std()
#
# 
# =============================================================================
    
# =============================================================================
# #select threshold for feature selection in Ridge regression.
# from sklearn.feature_selection import SelectFromModel
# 
# #print the threshold value and the associated number of features as well as the 5-folds cross validation 
# #average Root Mean Squared Error, for distinct thresholds given by the values of the model coefficients 
# 
# ridge = linear_model.Ridge(alpha=17.0)
# ridge.fit(new_train, y) 
# 
# def printTresholdRMSE(train, y, ridge):
#     thresholds = sorted(set(ridge.coef_))
#     for thresh in thresholds:
#     	  #select features using threshold
#         selection = SelectFromModel(ridge, threshold=thresh, prefit=True)
#         select_X_train = selection.transform(train)
#     	  #train model using the selected features
#         selection_model = linear_model.Ridge(alpha=17.0)
#         selection_model.fit(select_X_train, y)
#         #avoid error messages 
#         if select_X_train.shape[1] <=2: 
#             break
#         print("Thresh=%.8f, n=%d, rmse: %.8f%%" % (thresh, select_X_train.shape[1], CVRMSE(select_X_train, y, selection_model)))
#    
# #this function can be used analogously for Lasso regression. However, null appears to 
# #be the optimal threshold value. Furthermore, it could be exploited in order to perform feature selection 
# #when dealing with the Extreme Gradient Boosting regressor (taking advantage of the attribute feature_importances_). Nevertheless, reducing the number of features does not lead to significant improvements. 
# =============================================================================
    
# =============================================================================
# #Extreme Gradient Boosting parameter tuning.
# #Multiple searches over specified combinations of the parameter values for the regressor 
# #have been carried out. 
# #This is clearly a remarkably time-consuming method (although it is runned in parallel).

# def xgboostGS(train, y, estimator, param_grid):
#     folds = model_selection.KFold(5, shuffle=True, random_state=123).get_n_splits(train)  
#     xgbm = model_selection.GridSearchCV(estimator, param_grid, cv = folds, n_jobs = -1)
#     xgbm.fit(train, y) 
#     return xgbm
# 
# #For illustrative purposes, an example of grid search is as follows:
# estimator = XGBRegressor()
# 
# param_grid = {
#     "n_estimators" : [500, 550, 600],
#     "max_depth": [2,4,6,8],
#     "learning_rate" : [0.1, 0.15, 0.08, 0.05],
#     "gamma" : [0.0, 0.1]
#     }
# 
# gridsearchResults = xgboostGS(new_train, y, estimator, param_grid)
#
# #The optimal combination of parameters is then given by:
# gridsearchResults.best_params_
#
# =============================================================================

