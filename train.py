import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# -----------------------------
# Step 1: Column Names
# -----------------------------
columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

# Load dataset
train = pd.read_csv("KDDTrain+.txt", names=columns)

# Remove difficulty column
train = train.drop("difficulty", axis=1)

# Convert label to binary
train['label'] = train['label'].apply(lambda x: 0 if x == 'normal' else 1)

print("Preprocessing started...")

# -----------------------------
# Step 2: Encode Categorical Features
# -----------------------------
categorical_cols = ['protocol_type', 'service', 'flag']
le = LabelEncoder()

for col in categorical_cols:
    train[col] = le.fit_transform(train[col])

# -----------------------------
# Step 3: Split Features & Label
# -----------------------------
X = train.drop("label", axis=1)
y = train["label"]

# -----------------------------
# Step 4: Scale Features
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# Step 5: Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# Step 6: Train Random Forest
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Step 7: Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)

print("\nModel Training Complete ✅")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# Step 8: Save Model & Scaler
# -----------------------------
joblib.dump(model, "ids_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and Scaler saved successfully ✅")