import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

# Load the dataset
file_path = 'cleaned_data.csv'
data = pd.read_csv(file_path)

# Data Preprocessing
# Combine s:description and s:name as a single feature
X_simple = data['s:description'] + " " + data['s:name']

# Vectorize the combined text feature using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_simple_tfidf = tfidf_vectorizer.fit_transform(X_simple)

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['GS1_Level1_Category'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_simple_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Apply scaling
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to calculate and print classification report
def evaluate_model(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(f"Classification Report for {model_name}:\n", report_df)
    return report_df

# Store metrics for comparison
model_names = []
reports = {}

# Logistic Regression
logistic_model = LogisticRegression(max_iter=5000, solver='lbfgs', random_state=42)
logistic_model.fit(X_train_scaled, y_train)
logistic_predictions = logistic_model.predict(X_test_scaled)
logistic_report = evaluate_model(y_test, logistic_predictions, "Logistic Regression")
model_names.append('Logistic Regression')
reports['Logistic Regression'] = logistic_report

# Random Forest Model
random_forest_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
random_forest_model.fit(X_train, y_train)
random_forest_predictions = random_forest_model.predict(X_test)
random_forest_report = evaluate_model(y_test, random_forest_predictions, "Random Forest")
model_names.append('Random Forest')
reports['Random Forest'] = random_forest_report

# Gradient Boosting using XGBoost (if installed)
try:
    from xgboost import XGBClassifier
    gb_model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_predictions = gb_model.predict(X_test)
    gb_report = evaluate_model(y_test, gb_predictions, "XGBoost")
    model_names.append('XGBoost')
    reports['XGBoost'] = gb_report
except ImportError:
    print("XGBoost is not installed, falling back to GradientBoostingClassifier.")
    from sklearn.ensemble import GradientBoostingClassifier
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_predictions = gb_model.predict(X_test)
    gb_report = evaluate_model(y_test, gb_predictions, "Gradient Boosting")
    model_names.append('Gradient Boosting')
    reports['Gradient Boosting'] = gb_report

# Neural Network Model
nn_model = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=300, random_state=42)
nn_model.fit(X_train_scaled, y_train)
nn_predictions = nn_model.predict(X_test_scaled)
nn_report = evaluate_model(y_test, nn_predictions, "Neural Network")
model_names.append('Neural Network')
reports['Neural Network'] = nn_report

# Save the classification reports to CSV files
output_directory = 'output_results'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for model_name, report_df in reports.items():
    report_df.to_csv(os.path.join(output_directory, f'{model_name}_classification_report.csv'))

# Plotting line charts for each class metric
metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    plt.figure(figsize=(12, 8))
    for model_name in model_names:
        plt.plot(reports[model_name].index[:-3], reports[model_name][metric][:-3], marker='o', label=model_name)
        for i, value in enumerate(reports[model_name][metric][:-3]):
            plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')

    plt.xlabel('Class')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} Comparison Across Models')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f'{metric}_comparison_line_chart.png'))
    plt.show()
