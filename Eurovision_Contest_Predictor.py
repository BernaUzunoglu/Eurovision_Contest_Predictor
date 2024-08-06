import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import  StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib


# DANCEABILITY', 'ENERGY', 'LOUDNESS', 'SPEECHINESS', 'ACOUSTICNESS',
#        # 'LIVENESS', 'VALENCE', 'TEMPO', 'DURATION_MS', 'GDP_PER_CAPITA',
#        # 'SOCIAL_SUPPORT', 'HEALTHY_LIFE_EXPECTANCY_AT_BIRTH',
#        # 'FREEDOM_TO_MAKE_LIFE_CHOICES', 'GENEROSITY',
#        # 'PERCEPTIONS_OF_CORRUPTION', 'POSITIVE_AFFECT', 'NEGATIVE_AFFECT'],
def contest_predictor():
    data = pd.read_excel("datasets/Contest_Songs_Data.xlsx")
    df = data.drop(['year', 'artist_name', 'song_name', 'gender', 'country', 'area', 'Language', 'mode', 'key', 'time_signature', 'final_place', 'duration_ms', 'POSITIVE_AFFECT'], axis=1)
    df.columns = [col.upper() for col in df.columns]

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

    for col in df.columns:
        print(col, check_outlier(df, col))

    for col in df.columns:
        replace_with_thresholds(df, col)

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

    standatrlasancols = ['DANCEABILITY', 'ENERGY',
                         'LOUDNESS', 'SPEECHINESS', 'ACOUSTICNESS', 'INSTRUMENTALNESS',
                         'LIVENESS', 'VALENCE', 'TEMPO',
                         'GDP_PER_CAPITA', 'HEALTHY_LIFE_EXPECTANCY_AT_BIRTH']
    dff = num_cols_standardization(df, standatrlasancols, "ss")

    # Görev 3 : Model Kurma
    # Adım 1:  Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)
    ##################################################################################################
    y = data['final_place']
    X = df
    #############################################################################
    # Adım 2:  Train verisi ile model kurup, model başarısını değerlendiriniz.
    #############################################################################
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
    model = RandomForestRegressor()
    model.fit(X, y)

    # GridSearchCV
    # Hiperparametre aralığı
    param_grid = {
        'n_estimators': [80, 90, 100],  # Ağaç sayısı
        'max_features': ['auto', 'sqrt'],  # Özellik alt kümesi
        'max_depth': [1, None],  # Maksimum derinlik
        'min_samples_split': [9, 7, 11],  # Bir düğümdeki minimum örnek sayısı
        'min_samples_leaf': [5, 6, 7],  # Yaprak düğümdeki minimum örnek sayısı
        'bootstrap': [True, False]  # Örnekleme yöntemi
    }

    # Grid Search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=10, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)
    # Eğitim
    grid_search.fit(X_train, y_train)

    ##################

    # En iyi parametreleri al
    best_params = grid_search.best_params_
    print(f"En iyi hiperparametreler: {best_params}")
    # En iyi hiperparametrelerle modeli yeniden eğitme
    # best_model = RandomForestRegressor(**best_params)

    # En iyi model ile tahmin yap
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # RMSE hesapla
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # Modeli kaydetme
    joblib.dump(model, 'random_forest_eurovision_predictor.joblib')

    return best_model, rmse, data
