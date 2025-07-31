# Walmart Store Sales Forecasting - დროის სერიების პროგნოზირება

## პროექტის სტრუქტურა

```
walmart-sales-forecasting/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── notebooks/
│   ├── model_experiment_nbeats.ipynb
│   ├── model_experiment_tft.ipynb
│   ├── model_experiment_patchtst.ipynb
│   ├── model_experiment_dlinear.ipynb
│   ├── model_experiment_lightgbm.ipynb
│   ├── model_experiment_xgboost.ipynb
│   ├── model_experiment_arima.ipynb
│   ├── model_experiment_sarima.ipynb
│   ├── model_experiment_prophet.ipynb
│   └── model_inference.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── deep_learning.py
│   │   ├── tree_based.py
│   │   └── classical.py
│   └── utils.py
│
├── models/
│   └── trained_models/
│
├── submissions/
│
├── requirements.txt
└── README.md
```

აი თარგმანი ქართულად:

---

# EDA

## მონაცემების გამოკვლევის ანალიზი

მონაცემების გამოკვლევის ნოუთბუქი (`notebooks/data_exploration.ipynb`) გვთავაზობს Walmart-ის გაყიდვების მონაცემების ამომწურავ ანალიზს. ძირითადი მიგნებები მოიცავს:

### მონაცემების მიმოხილვა

* `train`, `test`, `stores` და `features` მონაცემთა სეტის ანალიზი
* მონაცემთა ტიპების, ნაკლული მნიშნელობების და საბაზისო სტატისტიკის შესწავლა
* დროითი დიაპაზონის დაფარვისა და მონაცემთა სრულობის შემოწმება

### ძირითადი ვიზუალიზაციები

* გაყიდვების დროითი გრაფიკები სხვადასხვა მაღაზიებსა და დეპარტამენტებზე
* გაყიდვების, ტემპერატურის, საწვავის ფასების და სხვა რიცხვითი მახასიათებლების განაწილების ანალიზი
* კორელაციის მატრიცები ცვლადებს შორის კავშირის გამოსავლენად
* ყუთის გრაფიკები (box plot) გაყიდვების განაწილებისა და ექსტრემალური მნიშვნელობების გასაანალიზებლად

### მახასიათებლების ანალიზი

* მაღაზიის ტიპების და მათი გაყიდვების შედეგიანობის შეფასება
* არდადეგებისა და სპეციალური ღონისძიებების გავლენა გაყიდვებზე
* ყოველკვირეული და ყოველთვიური გაყიდვების შაბლონების ანალიზი
* ფასდაკლებებისა და აქციების გავლენა გაყიდვებზე

### მონაცემთა ხარისხი

* დაკარგული მნიშნელობების და ექსტრემალური მონაცემების იდენტიფიკაცია
* მონაცემთა თანმიმდევრულობის ანალიზი სხვადასხვა წყაროს მიხედვით
* მონაცემთა შეყვანის პოტენციური შეცდომების გამოკვლევა

### დროითი რიგების კომპონენტები

* დროითი მონაცემების დekompozიცია ტენდენციად, სეზონურობად და ნაშთებად
* ყოველკვირეული და წლიური სეზონურობის ანალიზი
* დროით რიგებში სტრუქტურული ცვლილებების ან ანომალიების გამოვლენა

### სტატისტიკური ტესტები

* სტატიონარობის ტესტები დროით მონაცემებზე
* ნორმალურობის ტესტები ძირითადი ცვლადებისთვის
* კორელაციის ანალიზი მახასიათებლებსა და სამიზნე ცვლადს შორის

### ძირითადი მიგნებები

1. გაყიდვებში გამოირჩევა ძლიერი ყოველკვირეული და წლიური სეზონურობა
2. გაყიდვების მნიშვნელოვანი განსხვავებები სხვადასხვა მაღაზიის ტიპებსა და დეპარტამენტებს შორის
3. არდადეგებისა და აქციების აშკარა გავლენა გაყიდვებზე
4. გარე ფაქტორებს (ტემპერატურა, საწვავის ფასი) და გაყიდვებს შორის კავშირი
5. დროით შაბლონებზე დაფუძნებული ახალი მახასიათებლების შექმნის შესაძლებლობები
6. A ტიპის მაღაზიები სტაბილურად უკეთეს შედეგს აჩვენებენ, ვიდრე B და C ტიპები
7. დეპარტამენტების გაყიდვების ვარიაბელობა მაღალია — რამდენიმე დეპარტამენტი (მაგ., პროდუქტები, საყოველთაო საქონელი) დომინირებს საერთო გაყიდვებში, დანარჩენები კი ნიშურია
8. MarkDown აქციები იწვევს მოკლევადიან გაყიდვების მატებას, MarkDown1 და MarkDown4 ყველაზე ეფექტურია
9. თვის ბოლოს და ორკვირიან ხელფასის დღეებში გაყიდვების მკვეთრი მატებაა შესამჩნევი
10. მონაცემები სრულია — თარიღების გამოტოვება არ შეინიშნება; დაკარგული მნიშვნელობები ძირითადად markdown-ებსა და გარკვეულ მახასიათებლებში გვხვდება

### ვიზუალიზაციები მონაცემთა ანალიზიდან

* **საშუალო გაყიდვები მაღაზიის ტიპების მიხედვით:** სვეტების გრაფიკი, სადაც A ტიპი ლიდერობს გაყიდვებში, მოჰყვება B და C
* **გაყიდვები მაღაზიის ზომის მიხედვით:** ხაზოვანი გრაფიკი, რომელიც აჩვენებს რომ უფრო დიდი მაღაზიები, როგორც წესი, მეტ გაყიდვებს ახორციელებენ, თუმცა დიდი ზომის შემთხვევაში დადგება მომატების ზღვარი
* **ყველაზე ცვალებადი 20 დეპარტამენტი:** სვეტური გრაფიკი, რომელიც აჩვენებს რომელ დეპარტამენტებს აქვთ ყველაზე მაღალი გაყიდვების ვარიაბელობა
* **MarkDown აქციების გავლენა:** გრაფიკები, რომლებიც ადარებს საშუალო გაყიდვებს markdown-ის არსებობისა და არარსებობის დროს (მაგ., MarkDown1, MarkDown4)
* **დროითი გრაფიკები:** გაყიდვების ტენდენციები დროის მიხედვით შერჩეულ მაღაზიებსა და დეპარტამენტებში — სეზონურობისა და არდადეგების ეფექტის დემონსტრირებით
* **კორელაციის სითბური რუკები (heatmaps):** მახასიათებლებსა და გაყიდვებს შორის ურთიერთობის ვიზუალური წარმოდგენა (მაგ., ტემპერატურა, საწვავის ფასი, CPI, უმუშევრობა)

დეტალური კოდი და გრაფიკებისთვის იხილეთ `notebooks/data_exploration.ipynb`.


## MLflow ექსპერიმენტების სტრუქტურა

### ექსპერიმენტები:
1. **NBEATS_Training**
   - NBEATS_Cleaning
   - NBEATS_Feature_Engineering
   - NBEATS_Hyperparameter_Tuning
   - NBEATS_Cross_Validation

2. **TFT_Training**
   - TFT_Cleaning
   - TFT_Feature_Engineering
   - TFT_Hyperparameter_Tuning
   - TFT_Cross_Validation

3. **PatchTST_Training**
   - PatchTST_Cleaning
   - PatchTST_Feature_Engineering
   - PatchTST_Hyperparameter_Tuning
   - PatchTST_Cross_Validation

4. **DLinear_Training**
   - DLinear_Cleaning
   - DLinear_Feature_Engineering
   - DLinear_Hyperparameter_Tuning
   - DLinear_Cross_Validation

5. **LightGBM_Training**
   - LightGBM_Cleaning
   - LightGBM_Feature_Engineering
   - LightGBM_Hyperparameter_Tuning
   - LightGBM_Cross_Validation

6. **XGBoost_Training**
   - XGBoost_Cleaning
   - XGBoost_Feature_Engineering
   - XGBoost_Hyperparameter_Tuning
   - XGBoost_Cross_Validation

7. **ARIMA_Training**
   - ARIMA_Cleaning
   - ARIMA_Parameter_Selection
   - ARIMA_Cross_Validation

8. **SARIMA_Training**
   - SARIMA_Cleaning
   - SARIMA_Parameter_Selection
   - SARIMA_Cross_Validation

9. **Prophet_Training**
   - Prophet_Cleaning
   - Prophet_Feature_Engineering
   - Prophet_Hyperparameter_Tuning
   - Prophet_Cross_Validation

## მონაცემების ანალიზისა და მომზადების ეტაპები

### 1. Exploratory Data Analysis (EDA)
- მონაცემების სტრუქტურის შესწავლა
- დროის სერიების ვიზუალიზაცია
- სეზონურობისა და ტრენდების ანალიზი
- დასვენების დღეების გავლენის შესწავლა
- განყოფილებების შორის კორელაციის ანალიზი

### 2. Data Preprocessing
- ნაკლოვანი მნიშვნელობების დამუშავება
- გაუცნობელი ღირებულებების ფილტრაცია
- დროის ფორმატის სტანდარტიზაცია
- მონაცემების ნორმალიზაცია/სტანდარტიზაცია

### 3. Feature Engineering
- ლაგ ფიჩერების შექმნა
- მოძრავი საშუალოების დამატება
- სეზონური ფიჩერების განვითარება
- დასვენების დღეების კოდირება
- ტექნიკური ინდიკატორების გამოთვლა

## მოდელების დეტალური აღწერა

## Deep Learning მოდელები

### N-BEATS (Neural Basis Expansion Analysis for Time Series)
- **არქიტექტურა**: სტეკური სტრუქტურა ბლოკებით
- **უპირატესობები**: ინტერპრეტირებადობა, ტრენდი/სეზონურობის დეკომპოზიცია
- **განხორციელება**: PyTorch/TensorFlow
- **ჰიპერპარამეტრები**: stack რაოდენობა, ბლოკების რაოდენობა, თემატური/ჯენერიული


1. N-BEATS (Neural Basis Expansion Analysis for Time Series)
მოდელის არქიტექტურა
N-BEATS არის თანამედროვე ღრმა სწავლის მოდელი დროის მწკრივების პროგნოზირებისთვის, რომელიც:

სრულიად კონექტირებული ქსელი - არ ამოიყენებს CNN ან RNN-ს
მიმდევრობითი სტეკები - სამი სტეკი: Trend, Seasonality, Generic
Residual Connections - თითოეული სტეკი ასწორებს წინა სტეკის შეცდომებს
Forward/Backward მოდელირება - ბლოკები გენერირებენ როგორც პროგნოზს, ასევე ისტორიული მონაცემების რეკონსტრუქციას

შერჩეული ჰიპერპარამეტრები
ჰიპერპარამეტრების მოძებნის პროცესი
1. Input Size-ის ტიუნინგი:
input_size=40 → WMAE=1697.6167
input_size=52 → WMAE=1593.9089 ✓ (საუკეთესო)
input_size=60 → WMAE=1619.0920
input_size=72 → WMAE=1698.7422
2. Batch Size-ის ტიუნინგი:
batch_size=32  → WMAE=1619.6334
batch_size=64  → WMAE=1593.9089
batch_size=128 → WMAE=1586.4884
batch_size=256 → WMAE=1547.6261 ✓ (საუკეთესო)
batch_size=512 → WMAE=1581.5538
3. Learning Rate-ის ტიუნინგი:
learning_rate=0.001 → WMAE=1547.6261 ✓ (საუკეთესო)
learning_rate=0.002 → WMAE=1578.8466
learning_rate=0.004 → WMAE=1590.2231
4. ბლოკების რაოდენობა და Weight Decay:
n_blocks=[1,1,1] + weight_decay=0.0001 → WMAE=1563.2994 ✓ (საუკეთესო)
n_blocks=[2,2,2] + weight_decay=0.0001 → WMAE=1602.7405
n_blocks=[3,3,3] + weight_decay=0.0001 → WMAE=1615.8575
5. ღიმის ფუნქციების შედარება:
activation=LeakyReLU → WMAE=1579.8415 ✓ (საუკეთესო)
activation=ReLU     → WMAE=1587.5859
activation=Tanh     → WMAE=2986.5779
activation=PReLU    → WMAE=1620.2018
საბოლოო კონფიგურაცია
pythonmodel = NBEATS(
    max_steps=2600,        # 25 * 104 epoch
    h=53,                  # პროგნოზის ჰორიზონტი
    input_size=52,         # შეყვანის ფანჯრის ზომა
    batch_size=256,        # batch size
    learning_rate=1e-3,    # სწავლის სიჩქარე
    shared_weights=True,   # ზოგადი წონები
    optimizer=torch.optim.AdamW,
    activation='ReLU',     # ღუმის ფუნქცია
    n_blocks=[1,1,1],     # თითო სტეკში ბლოკების რაოდენობა
    random_seed=42
)
მოდელის უპირატესობები

ინტერპრეტირებადობა: თითოეული სტეკი განსაზღვრულ კომპონენტს აანალიზებს
მაღალი წარმადობა: კარგად მუშაობს გრძელვადიან პროგნოზირებაზე
ღუმის კარების გარეშე: უფრო სტაბილური ტრენინგი

მონაცემების პროცესინგი
1. მონაცემების ჩატვირთვა
pythonstores = pd.read_csv('data/stores.csv')
features = pd.read_csv('data/features.csv')
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
2. პრე-პროცესინგი

თარიღის კონვერტაცია: pd.to_datetime()
უნიკალური ID-ს შექმნა: Store + Dept
ნაკლული მნიშვნელობების შევსება: fillna(0)
მონაცემების დალაგება: Store, Dept, Date-ით

3. ფიჩერების ინჟინერია

Calendar ფიჩერები: კვირის დღე, თვე, წელი
Lag ფიჩერები: წინა კვირების გაყიდვები
მოძრავი საშუალო: 4, 8, 12 კვირის მავალი საშუალო
სტატისტიკური ფიჩერები: min, max, std rollling windows

4. დაყოფა და ვალიდაცია

Train/Validation Split: 80/20
Time-based Split: ბოლო 20% თარიღებით
Cross-Validation: 3-fold time series CV



# Wandb-ში ატვირთვა
artifact = wandb.Artifact(name="nbeats_final_model", type="model")
artifact.add_file("nbeats_final_model.pkl")
wandb.log_artifact(artifact)
შედეგების ანალიზი
მოდელის შეფასება
მეტრიკები

WMAE: 1547.6261
MAE: 1423.45
RMSE: 2134.67

შეცდომის ანალიზი

სეზონურ პერიოდებში: მოდელი კარგად იცნობს სეზონურ პატერნებს
სადღესასწაულო კვირებში: WMAE-ის გამო მნიშვნელოვანი მხედველობაშია
გამოუყენებელი მაღაზიები: ახალი მაღაზიების პროგნოზირება რთულია



### Temporal Fusion Transformer (TFT)
- **არქიტექტურა**: Attention მექანიზმი დროის სერიებისთვის
- **უპირატესობები**: მრავალ-ცვლადიანი, ინტერპრეტირებადი attention
- **განხორციელება**: PyTorch Forecasting
- **ჰიპერპარამეტრები**: attention heads, hidden dimensions, dropout

### PatchTST
- **არქიტექტურა**: Vision Transformer-ის ადაპტაცია
- **უპირატესობები**: პეჩებად დაყოფა, ეფექტური ტრენინგი
- **განხორციელება**: HuggingFace Transformers
- **ჰიპერპარამეტრები**: patch size, model dimensions, layers

### DLinear
- **არქიტექტურა**: ბოლო კვლევების მარტივი მიდგომა
- **უპირატესობები**: სიმარტივე, სისწრაფე
- **განხორციელება**: PyTorch
- **ჰიპერპარამეტრები**: kernel size, channels

## Tree-Based მოდელები

### LightGBM
- **უპირატესობები**: სისწრაფე, მეხსიერების ეფექტურობა
- **ფიჩერ ინჯინირინგი**: ლაგები, სტატისტიკური ფიჩერები
- **ჰიპერპარამეტრები**: num_leaves, learning_rate, feature_fraction

LightGBM მოდელის ექსპერიმენტი
1. მონაცემთა წინასწარი დამუშავება (Data Preprocessing)
განხორციელებული ნაბიჯები:

მონაცემთა შერწყმა: ყველა ფაილის გაერთიანება
თარიღის ფიჩები: წელი, თვე, კვირა, დღე, კვირის დღე
ციკლური ფიჩები: sin/cos ტრანსფორმაცია თვისა და კვირისთვის
კატეგორიული ცვლადების კოდირება: LabelEncoder-ის გამოყენება
რიცხვითი ცვლადების სტანდარტიზება: StandardScaler-ის გამოყენება

შედეგები:

სატრენინგო მონაცემები: ~421,000 ნიმუში
ტესტის მონაცემები: ~115,000 ნიმუში
ფიჩების საერთო რაოდენობა: 25+

2. ფიჩების ინჟინერინგი (Feature Engineering)
შექმნილი ფიჩები:

Lag Features: წინა 1, 2, 4, 8, 12 კვირის გაყიდვები
Rolling Statistics: 4, 8, 12 კვირის მოძრავი საშუალო და სტანდარტული გადახრა
Store-Department Statistics: საშუალო, სტანდარტული გადახრა, მინ/მაქს გაყიდვები
Time-based Features: თვე, კვირა, კვირის დღე, არის კი უქმე დღე
Holiday Features: საზღვარბათო დღეების ინდიკატორი

ფიჩების მნიშვნელობა:
ფიჩების შერჩევისთვის გამოყენებულია LightGBM-ის feature importance, სადაც აღმოჩნდა:

Dept_Sales_Mean - უმაღლესი მნიშვნელობა
Store - მაღაზიის იდენტიფიკატორი
Size - მაღაზიის ზომა
Sales_Lag_1 - წინა კვირის გაყიდვები
Temperature - ტემპერატურა

3. მოდელის ტრენინგი და ვალიდაცია
გამოყენებული მიდგომა:

Time Series Cross-Validation: 5-ფოლდით განაწილება
Evaluation Metric: RMSE (Root Mean Squared Error)
Competition Metric: Weighted MAE (საზღვარბათო კვირები x5 წონა)

LightGBM პარამეტრები:
```
pythonlgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'n_estimators': 1000
}
```
#### შედეგები:

Cross-Validation RMSE: ~12,500
Cross-Validation WMAE: ~8,200
Final Training RMSE: ~11,800
Final Training MAE: ~7,900

### XGBoost
- **უპირატესობები**: მტკიცებულება, regularization
- **ფიჩერ ინჯინირინგი**: ანალოგიური LightGBM-თან
- **ჰიპერპარამეტრები**: max_depth, eta, subsample

### Classical მოდელები

#### ARIMA/SARIMA
- **მიდგომა**: ავტორეგრესიულ ინტეგრირებული მოძრავი საშუალო მოდელი დროის სერიებისთვის
- **პარამეტრები**:
  - ARIMA: p=3, d=1, q=2 (მორგებული ACF/PACF ანალიზით)
  - SARIMA: (P=1, D=1, Q=1, s=52) წლიური სეზონურობისთვის
- **სტაციონარობის ტესტები**:
  - ADF p-value: 0.01 (p<0.05, სტაციონარული)
  - KPSS p-value: 0.1 (>0.05, სტაციონარული)
- **შედეგები**:
  - Training MAE: 2150.45
  - Validation MAE: 2280.67
  - AIC: 12500.32
  - BIC: 12580.15

#### Prophet
- **უპირატესობები**: სეზონურობის ავტომატური გამოვლენა
- **ფიჩერები**: დასვენების დღეები, ტრენდის ცვლილებები
- **ჰიპერპარამეტრები**: seasonality parameters, holidays

## Cross-Validation სტრატეგია

### Time Series Cross-Validation
- **მიდგომა**: Forward chaining / Rolling window
- **ვალიდაციის ფანჯრები**: 4-5 ფოლდი
- **ტესტის მონაცემები**: ბოლო 8 კვირა

## შეფასების მეტრიკები

### მთავარი მეტრიკა: WMAE (Weighted Mean Absolute Error)
```python
def wmae(y_true, y_pred, weights):
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)
```

### დამატებითი მეტრიკები:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- SMAPE (Symmetric Mean Absolute Percentage Error)

## Pipeline და Model Registry

### Pipeline კომპონენტები:
1. **Data Preprocessing Pipeline**
2. **Feature Engineering Pipeline**
3. **Model Training Pipeline**
4. **Prediction Pipeline**

### Model Registry:
- საუკეთესო მოდელის რეგისტრაცია
- ვერსიების მართვა
- A/B ტესტირების მხარდაჭერა

## ჩასატარებელი ექსპერიმენტები

### Feature Engineering:
- ლაგ ფიჩერები (1, 2, 4, 8, 12, 52 კვირა)
- მოძრავი საშუალოები (4, 8, 12, 26 კვირა)
- სეზონური დეკომპოზიცია
- პრაისის ელასტიურობა
- კონკურენტული ანალიზი

### Model Ensembling:
- Weighted averaging
- Stacking
- Blending
- Dynamic ensemble selection
