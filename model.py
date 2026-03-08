import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- STEP 1: LOAD DATA ---
folder_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(folder_path, 'data.csv')

try:
    df = pd.read_csv(data_path)
    print("✅ Data loaded successfully!")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit()

# --- STEP 2: EDA (Exploratory Data Analysis) ---
print("\n--- Performing EDA ---")
# Plot Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, palette='viridis')
plt.title('Target Class Distribution (Toxic vs Non-Toxic)')
plt.savefig(os.path.join(folder_path, 'eda_class_distribution.png'))
print("📊 Saved: eda_class_distribution.png")

# --- STEP 3: PREPROCESSING ---
# Target encoding
df['Class_mapped'] = df['Class'].map({'NonToxic': 0, 'Toxic': 1})
X = df.drop(columns=['Class', 'Class_mapped'])
y = df['Class_mapped']

# Remove constant/near-constant features (Variance Thresholding)
X = X.loc[:, X.var() > 0.01]

# Remove highly correlated features (> 0.95) to reduce redundancy
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_reduced = X.drop(columns=to_drop)

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)

# --- STEP 4: FEATURE SELECTION ---
# Use Random Forest to select the top 15 most important features
selector_model = RandomForestClassifier(n_estimators=100, random_state=42)
selector = SelectFromModel(selector_model, max_features=15)
X_selected = selector.fit_transform(X_scaled, y)
selected_names = X_reduced.columns[selector.get_support()]
print(f"✅ Feature Selection Complete. Selected: {len(selected_names)} descriptors.")

# --- STEP 5: MODEL TRAINING & CROSS-VALIDATION ---
# Balanced class weight helps if you have more Non-Toxic than Toxic samples
model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Calculate Cross-Validation Scores
acc_scores = cross_val_score(model, X_selected, y, cv=cv, scoring='accuracy')
f1_scores = cross_val_score(model, X_selected, y, cv=cv, scoring='f1')

print("\n--- MODEL PERFORMANCE (Cross-Validation) ---")
print(f"Mean Accuracy: {acc_scores.mean():.2%}")
print(f"Mean F1-Score: {f1_scores.mean():.4f}")

# --- STEP 6: FINAL VISUALS ---
model.fit(X_selected, y)

# 1. Feature Importance Plot
plt.figure(figsize=(10, 6))
importances = pd.Series(model.feature_importances_, index=selected_names).sort_values(ascending=True)
importances.plot(kind='barh', color='teal')
plt.title('Top Predictive Molecular Descriptors')
plt.tight_layout()
plt.savefig(os.path.join(folder_path, 'feature_importance.png'))

# 2. Confusion Matrix
y_pred = model.predict(X_selected)
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NonToxic', 'Toxic'])
disp.plot(cmap='Blues', ax=ax)
plt.title('Model Confusion Matrix')
plt.savefig(os.path.join(folder_path, 'confusion_matrix.png'))

# 3. Heatmap of Selected Features
plt.figure(figsize=(12, 10))
sns.heatmap(X_reduced[selected_names].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(folder_path, 'feature_heatmap.png'))

print("\n🚀 ALL STEPS COMPLETE. Check your folder for the 4 image results!")