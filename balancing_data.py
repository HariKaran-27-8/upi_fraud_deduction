import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Load your dataset
df = pd.read_csv("upi_data.csv")

# Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df.drop(columns=['Date'], inplace=True)

# Separate classes
df_majority = df[df.fraud == 0]
df_minority = df[df.fraud == 1]

print("Before balancing:")
print(df['fraud'].value_counts())

### --------------------
### OPTION 1: UNDERSAMPLING
### --------------------
# df_majority_downsampled = resample(df_majority, 
#                                    replace=False,    
#                                    n_samples=len(df_minority),     
#                                    random_state=42)

# df_balanced = pd.concat([df_majority_downsampled, df_minority])
# df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)

### --------------------
### OPTION 2: SMOTE (Recommended)
### --------------------
from sklearn.preprocessing import OneHotEncoder

# Separate features and target
X_raw = df.drop(columns=['fraud'])
y = df['fraud']

# One-hot encode categorical
cat_cols = ['Transaction_Type', 'Payment_Gateway', 'Transaction_State', 'Merchant_Category']
num_cols = ['amount', 'Month', 'Year']

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(X_raw[cat_cols])
X_all = pd.concat([X_raw[num_cols].reset_index(drop=True), pd.DataFrame(X_cat)], axis=1)
X_all.columns = X_all.columns.astype(str)
# Apply SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_all, y)

print("After SMOTE:")
print(pd.Series(y_res).value_counts())

# Save to CSV if needed
# X_res['fraud'] = y_res
# X_res.to_csv("balanced_upi_data.csv", index=False)
# ...existing code...

# After SMOTE, convert X_res to DataFrame
feature_names = num_cols + list(encoder.get_feature_names_out(cat_cols))
X_res_df = pd.DataFrame(X_res, columns=feature_names)
X_res_df['fraud'] = y_res
X_res_df.to_csv("balanced_upi_data.csv", index=False)
