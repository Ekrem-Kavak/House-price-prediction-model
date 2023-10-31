# HOUSE-PRICE PREDICTION MODEL (EV-FİYAT TAHMİN MODELİ)

"""
Her bir eve ait özelliklerin ve ev fiyatlarının bulunduğu veri seti
kullanılarak, farklı tipteki evlerin fiyatlarına ilişkin bir makine
öğrenmesi projesi gerçekleştirilmek istenmektedir.
İlgili veri setinde;
1460 gözlem,
38 sayısal değişken,
43 kategorik değişken bulunmaktadır.

"""

# EXPLORATORY DATA ANALYSIS (KEŞİFÇİ VERİ ANALİZİ)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# GÖREV-1: Train ve test setlerini okutup birleştiriniz. Birleştirdiğiniz veri üzerinden ilerleyiniz.

train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")
df = train.append(test, ignore_index = False).reset_index()
df.head()

# Sırası belli olduğu için "index" değişkenini silelim.

df = df.drop("index", axis = 1)

# GÖREV-2: Veri seti hakkında genel özelliklerini inceleyin.

def check_df(dataframe):
    print("  shape  ")
    print(dataframe.shape)
    print("  dtypes  ")
    print(dataframe.dtypes)
    print("  ilk 5 satır  ")
    print(dataframe.head())
    print("  son 5 satır  ")
    print(dataframe.tail())
    print("  NA  ")
    print(dataframe.isnull().sum())
    print("  Quantiles  ")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# GÖREV-3: Numerik ve kategorik değişkenlerin veri seti içindeki dağılımını gözlemleyiniz.

def grab_col_names(dataframe, cat_th = 10, car_th = 20):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variable: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_car: {len(num_but_cat)}")

    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)

# GÖREV-4: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

# kategorik
def cat_summary(dataframe, col_name, plot = True):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x = dataframe[col_name], data = dataframe)
        plt.show(block = True)

    for col in cat_cols:
        cat_summary(df, col)

# numerik
def num_summary(dataframe, numerical_col, plot = True):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99, 1]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block = True)

for col in num_cols:
    num_summary(df, col, True)

# GÖREV-5: Kategorik değişkenler ile hedef değişkenler incelemesini yapınız.

# SalePrice değiğşkeninin kategorik değişkenler ile ilişkisi

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end = "\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "SalePrice", col)


# Bağımlı değiğşkenin logaritmasının incelenmesi
np.log1p(df["SalePrice"]).hist(bins = 50)
plt.show(block = True)

# Korelasyon analizi
corr = df[num_cols].corr()

sns.set(rc = {'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap = "RdBu")

def high_corralated_cols(dataframe, plot = False, corr_th = 0.70):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show(block = True)
    return drop_list

high_corralated_cols(df, plot = True)

# GÖREV-6: Aykırı gözlem var mı inceleyiniz.

# aykırı değerlerin tespit edilmesi
def outlier_thresholds(dataframe, variable, low_quantile = 0.10, up_quantile = 0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis = None):
        return True
    else:
        return False

for col in num_cols:
    if col != "SalePrice":
        print(col, check_outlier(df, col))

# aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df, col)

# GÖREV-7: Eksik gözlem var mı inceleyiniz.




def missing_values_table(dataframe, na_name = False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending = False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending = False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis = 1, keys = ['n_miss', 'ratio'])

    print(missing_df, end = "\n")

    if na_name:
        return na_columns

missing_values_table(df)

# Bazı değişkenlerdeki boş değerler ilgili evin o özelliğe sahip olmadığını gösterir.
no_cols = ["Alley", "BsmtQual", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
           "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "PoolQC", "Fence", "MiscFeature"]

# Kolonlardaki boşlukların "No" ifadesi ile doldurulması
for col in no_cols:
    df[col].fillna("No", inplace = True)

missing_values_table(df)
