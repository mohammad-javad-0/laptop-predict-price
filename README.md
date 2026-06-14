# Laptop Price Prediction Pipeline



An end-to-end Machine Learning project designed to predict laptop prices using advanced data preprocessing, feature engineering, and optimized regression models.



This repository covers the complete machine learning workflow, including Exploratory Data Analysis (EDA), custom feature extraction from raw specifications, preprocessing pipelines, model benchmarking, hyperparameter tuning, and final model deployment.



---



# Project Architecture & Workflow



The project is divided into two main components to ensure a clean and reproducible machine learning workflow:



1. **Exploratory Data Analysis (`eda.ipynb`)**

&#x20;  - Dataset investigation

&#x20;  - Feature extraction

&#x20;  - Data cleaning

&#x20;  - Statistical analysis

&#x20;  - Outlier analysis



2. **Model Training Pipeline (`modeling.ipynb`)**

&#x20;  - Data preprocessing

&#x20;  - Model comparison

&#x20;  - Hyperparameter optimization

&#x20;  - Final model training

&#x20;  - Model serialization



---



# Dataset Attribution



The dataset used in this project was obtained from Kaggle:



**Dataset Source:**  

https://www.kaggle.com/datasets/muhammetvarl/laptop-price



The dataset contains laptop specifications and their corresponding prices in Euros. It includes information about laptop brands, processors, GPUs, RAM, storage, screen specifications, operating systems, weight, and price.



This dataset was used for educational and machine learning purposes, including exploratory data analysis, feature engineering, preprocessing, and regression modeling.



Please refer to the original Kaggle dataset page for the dataset license and usage terms.



Features include:



- Laptop company

- Product name

- Laptop type

- CPU information

- GPU information

- RAM

- Storage

- Screen specifications

- Operating system

- Weight

- Price



The original dataset file used in this project:



```

laptop_price.csv

```



---



# 1. Exploratory Data Analysis (EDA) & Feature Engineering



During the EDA phase, the raw laptop specification data was transformed into meaningful numerical features.



## CPU Feature Extraction



The original `Cpu` column contained complex string values.



Extracted features:



- `Cpu_brand`

- `Cpu_model`

- `Cpu_clock_rate_(GHz)`



---



## GPU Feature Extraction



The original `Gpu` column was processed to extract:



- `Gpu_brand`

- `Gpu_series`



---



## Screen Feature Engineering



The `ScreenResolution` column was transformed into:



- `Screen width resolution`

- `Screen height resolution`

- `IPS panel indicator`

- `Touchscreen indicator`

- `Retina display indicator`



A new feature was created:



```

PPI (Pixels Per Inch)

```



to represent display density.



---



## RAM Feature Extraction



The original `RAM` string values were converted into numerical format:



```

Ram_(GB)

```



---



## Weight Feature Extraction



The `weight` column was cleaned and converted into:



```

Weight_(kg)

```



---



## Storage Feature Engineering



The original `memory` information was decomposed into separate features:



- `SSD capacity`

- `HDD capacity`

- `Flash Storage capacity`

- `Hybrid storage capacity`



Generated features:



```

Memory_SSD_(GB)

Memory_HDD_(GB)

Memory_Flash_Storage_(GB)

Memory_Hybrid_(GB)

```



---



# Missing Value Analysis



The dataset was checked for missing values.



Result:



```

No missing values were found.

```



---



# Data Distribution Analysis



The target variable:



```

Price_euros

```



was highly right-skewed.



To improve model performance, a logarithmic transformation was applied:



```

log1p(target)

```



During evaluation, predictions were converted back using:



```

expm1()

```



using `TransformedTargetRegressor`.



---



# Outlier Analysis



Outliers were analyzed using the IQR method.



Features analyzed:



- Price

- RAM

- SSD capacity

- PPI

- CPU clock rate



After investigation, extreme values were kept because they represented real high-end laptops rather than incorrect data.



---



# 2. Data Preprocessing & Pipeline Construction



To prevent **Data Leakage**, preprocessing steps were integrated inside Scikit-Learn pipelines.



Two preprocessing strategies were used.



---



## Pipeline for Linear & Distance-Based Models



Models:



- Linear Regression

- ElasticNet

- SVR





### Categorical Features



```

OneHotEncoder(handle_unknown='ignore')

```



### Numerical Features



```

RobustScaler()

      ↓

PowerTransformer(method='yeo-johnson')

```



---



## Pipeline for Tree-Based Models



Models:



- Random Forest

- Gradient Boosting

- AdaBoost





Tree models used:



```

OneHotEncoder()

```



for categorical features and passed numerical features without scaling.



---



# 3. Model Training & Evaluation



The training process consisted of two stages.



---



# Phase 1: Base Model Benchmarking



Six regression models were evaluated using 5-Fold Cross Validation:



| Model | Status |

| Random Forest Regressor | Candidate Selected |

| Gradient Boosting Regressor | Candidate Selected |

| Support Vector Regressor (SVR) | Candidate Selected |

| AdaBoost Regressor | Excluded |

| ElasticNet | Excluded |

| Linear Regression | Evaluated |



Evaluation metrics:



- R² Score

- MAE

- RMSE



---



# Phase 2: Hyperparameter Tuning



The selected models:



- SVR

- Random Forest

- Gradient Boosting



were optimized using:



```

RandomizedSearchCV

```



with cross-validation.



The SVR model achieved the best RMSE performance and was selected as the final model.



---



# Final Model



The final model:



```

Support Vector Regressor (SVR)

```



Configuration:



```python

{
   'kernel': 'rbf',
   'C': 3.593813663804626,
   'gamma': 0.01,
   'degree': 4
}

```



The final pipeline was trained on the complete dataset and saved:



```

final_model/final_model.pkl

```



---



# How to Run & Reproduce



## 1. Clone Repository



```bash

git clone https://github.com/mohammad-javad-0/laptop-predict-price.git



cd laptop-price-prediction

```



---



## 2. Install Dependencies



```bash

pip install -r requirements.txt

```



---



## 3. Dataset Setup



Download the dataset from Kaggle:



```

https://www.kaggle.com/datasets/muhammetvarl/laptop-price

```



Place the file:



```

laptop_price.csv

```



inside:



```

data/laptop_price.csv

```



---



## 4. Execute Notebooks



Run:



```

source_code/eda.ipynb

```



for data analysis and feature engineering.



Then run:



```

source_code/modeling.ipynb

```



for model training, tuning, and final model export.



---

