from preprocessing import load_and_preprocess
from decisionTree import DecisionTree
from randomForest import train_random_forest
from sklearn.metrics import accuracy_score, classification_report


X_train, X_test, y_train, y_test = load_and_preprocess("AER_credit_card_data.csv")



print("=== Decision Tree (Scratch) ===")
dt = DecisionTree(max_depth=5)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))


print("\n=== Random Forest (Sklearn) ===")
rf = train_random_forest(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
