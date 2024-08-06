import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
import warnings
import mpmath as rf
from sklearn.tree import export_text

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 900)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_excel("datasets/Contest_Songs_Data.xlsx")
df.columns = [col.upper() for col in df.columns]

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outlier(df, col))


for col in num_cols:
    replace_with_thresholds(df, col)


cat_cols = [col for col in cat_cols if "FINAL_PLACE" not in col]
num_cols.append('FINAL_PLACE')

##################################################
#  Encoding işlemlerini gerçekleştiriniz.
##################################################
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


dff2 = rare_analyser(df,'FINAL_PLACE',['LANGUAGE'])

dff = rare_encoder(df, 0.011)
dff=dff.drop(["YEAR","COUNTRY",'ARTIST_NAME','SONG_NAME', "DURATION_MS","POSITIVE_AFFECT"], axis=1)
dff.columns
dff['AREA'].value_counts()
dff.head(20)
onehotcols=['GENDER', 'MODE', 'TIME_SIGNATURE','LANGUAGE','KEY','AREA']
def one_hot_encoder(dataframe, onehotcols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=onehotcols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, onehotcols, drop_first=True)
type(dff)
dff.head()
df['LANGUAGE'].value_counts()

############################################################
# Numerik değişkenler için standartlaştırma yapınız.
#############################################################
def num_cols_standardization(dataframe, num_cols, scaler_type="ss"):
    """
    Verilen veriye göre belirtilen ölçekleyici tipini uygulayan bir fonksiyon.

    Args:
    scaler_type (str): Kullanılacak ölçekleyici tipi ('ss : Standart Scaler', 'rs : RobustScaler', 'mms : MinMax Scaler').
    data (pandas.DataFrame): Ölçeklemek istediğimiz veri seti.

    Returns:
    pandas.DataFrame: Ölçeklenmiş veri seti.
    """
    if scaler_type == 'ss':
        scaler = StandardScaler()
    elif scaler_type == 'rs':
        scaler = RobustScaler()
    elif scaler_type == 'mms':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Geçersiz ölçekleyici tipi. 'ss', 'rs' veya 'mms' olmalıdır.")

    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
    return dataframe
ss = StandardScaler()

standatrlasancols=['DANCEABILITY', 'ENERGY',
       'LOUDNESS', 'SPEECHINESS', 'ACOUSTICNESS', 'INSTRUMENTALNESS',
       'LIVENESS', 'VALENCE', 'TEMPO',
       'GDP_PER_CAPITA','HEALTHY_LIFE_EXPECTANCY_AT_BIRTH']
dff.columns
dff = num_cols_standardization(dff, standatrlasancols, "ss")
dff.head()
dff.shape
type(dff)


# Görev 3 : Model Kurma
# Adım 1:  Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)
##################################################################################################
y = df['FINAL_PLACE']
X = dff.drop(["FINAL_PLACE"], axis=1)
X.head(1)
y.head(1)


#############################################################################
# Adım 2:  Train verisi ile model kurup, model başarısını değerlendiriniz.
#############################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)



models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]
          # ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


df['FINAL_PLACE'].mean()
df['FINAL_PLACE'].std()

# RMSE: 13.7191 (LR)
# RMSE: 13.2963 (Ridge)
# RMSE: 13.8461 (Lasso)
# RMSE: 13.9603 (ElasticNet)
# RMSE: 15.4102 (KNN)
# RMSE: 18.4333 (CART)
# RMSE: 13.098 (RF)
# RMSE: 14.2313 (SVR)
# RMSE: 13.3676 (GBM)
# RMSE: 13.871 (XGBoost)
# RMSE: 13.6023 (LightGBM)



model = RandomForestRegressor()
model.fit(X, y)

predictions = model.predict(dff.drop(["FINAL_PLACE"], axis=1))

dictionary = {"Row":df.index, "FINAL_PLACE":predictions}

dfSubmission = pd.DataFrame(dictionary)
dfSubmission = pd.concat([dfSubmission, df["FINAL_PLACE"]], axis=1)
##################################################################
################################################################
# Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.
################################################################

# feature importance
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

plot_importance(model, X)
# Random Forest modelinden alınan özellik önemlilikleri
importances = model.feature_importances_
features = X.columns
# Eşik değer (örnek olarak 0.02 kullanıyoruz)
threshold = 0.025

# Eşik değerinin üzerindeki özellikleri seçin
important_features = features[importances > threshold]
X_reduced = X[important_features]
X=X_reduced
X.columns
#####################################

model = RandomForestRegressor()
model.fit(X, y)

predictions = model.predict(dff[X.columns])

dictionary = {"Row":df.index, "FINAL_PLACE":predictions}

dfSubmission = pd.DataFrame(dictionary)
dfSubmission = pd.concat([dfSubmission, df["FINAL_PLACE"]], axis=1)
dfSubmission
# GridSearchCV
# Hiperparametre aralığı
param_grid = {
    'n_estimators': [70,80,90,100],     # Ağaç sayısı
    'max_features': ['auto', 'sqrt'],      # Özellik alt kümesi
    'max_depth': [1, None],          # Maksimum derinlik
    'min_samples_split': [9],       # Bir düğümdeki minimum örnek sayısı
    'min_samples_leaf': [1],         # Yaprak düğümdeki minimum örnek sayısı
    'bootstrap': [True, False]             # Örnekleme yöntemi
}

# Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)

# Eğitim
grid_search.fit(X_train, y_train)

##################

# En iyi parametreleri al
best_params = grid_search.best_params_
print(f"En iyi hiperparametreler: {best_params}")

# En iyi model ile tahmin yap
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# RMSE hesapla
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")

##########tekrar tahmin ettirme#######
predictions = best_model.predict(dff.drop(["FINAL_PLACE"], axis=1))

dictionary = {"Row":df.index, "seda_predict":predictions}

dfSubmission = pd.DataFrame(dictionary)
dfSubmission = pd.concat([dfSubmission, df["FINAL_PLACE"]], axis=1)

dfSubmission.loc[77]
df.loc[75:85]


