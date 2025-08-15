# 💳 Credit Score & Risk Predictor

This is an interactive web application built with Streamlit that predicts a user's credit score and their probability of defaulting on a loan. The application uses a machine learning model trained on financial data to provide these estimates.

In addition to providing a score, the app also visualizes how the user's risk profile compares to others in their same age and income bracket, offering valuable context to the prediction.

---

### Prerequisites to run locally
- Git LFS installed

---

## 📂 File Structure
├── app.py                      # The main Streamlit application script
├── best_credit_score_model.joblib  # The trained machine learning model
├── imputer.joblib              # The saved imputer for handling missing values
├── scaler.joblib               # The saved feature scaler
├── cs-test-modified.csv        # The dataset used for the comparison plot
├── .gitattributes              # Configures Git LFS to track large files
├── README.md                   # This file
└── .gitignore                  # Specifies which files Git should ignore  


---

## 🤖 Model Details

The prediction is powered by a **Gradient Boosting Classifier**, which was chosen for its high accuracy on this dataset. The model was trained on the "Give Me Some Credit" dataset from Kaggle. The training process involved:

---

## 🛠️ Technologies Used

- **Streamlit:** For building the interactive web application.
- **Pandas:** For data manipulation and analysis.
- **Scikit-learn:** For the machine learning model, imputation, and scaling.
- **Matplotlib:** For creating the comparison box plot.
- **Git & Git LFS:** For version control and managing large files.
