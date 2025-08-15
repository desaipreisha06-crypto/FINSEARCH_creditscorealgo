# 💳 Credit Score & Risk Predictor

This is an interactive web application built with Streamlit that predicts a user's credit score and their probability of defaulting on a loan. The application uses a machine learning model trained on financial data to provide these estimates.

In addition to providing a score, the app also visualizes how the user's risk profile compares to others in their same age and income bracket, offering valuable context to the prediction.

![App Screenshot](https://i.imgur.com/your-screenshot-url.png) <!-- Replace with a URL to a screenshot of your app -->

---

## ✨ Features

- **Interactive Score Prediction:** Enter your financial details into a clean, user-friendly form.
- **Instant Results:** Get an estimated credit score and probability of default in real-time.
- **Comparative Analysis:** See a box plot that shows how your default risk compares to a peer group with a similar age and income.
- **Detailed Statistics:** View descriptive statistics (mean, median, quartiles) for your comparison group.
- **Light & Dark Mode:** The application is styled for both light and dark viewing modes.

---

## 🚀 How to Run Locally

To run this application on your local machine, please follow these steps.

### Prerequisites

- Python 3.8+
- Git and Git LFS installed

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install Git LFS files:**
    ```bash
    git lfs pull
    ```

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install the required libraries:**
    ```bash
    pip install streamlit pandas scikit-learn matplotlib
    ```

### Running the App

1.  **Launch the Streamlit server:**
    ```bash
    streamlit run app.py
    ```
2.  Open your web browser and navigate to the local URL provided in the terminal (usually `http://localhost:8501`).

---

## 📂 File Structure


├── app.py                      # The main Streamlit application script
├── best_credit_score_model.joblib  # The trained machine learning model
├── imputer.joblib              # The saved imputer for handling missing values
├── scaler.joblib               # The saved feature scaler
├── cs-test-modified.csv        # The dataset used for the comparison plot
├── .gitignore                  # Specifies which files Git should ignore
├── .gitattributes              # Configures Git LFS to track large files
└── README.md                   # This file


---

## 🤖 Model Details

The prediction is powered by a **Gradient Boosting Classifier**, which was chosen for its high accuracy on this dataset. The model was trained on the "Give Me Some Credit" dataset from Kaggle. The training process involved:
1.  Imputing missing values using the mean.
2.  Scaling features using `StandardScaler`.
3.  Training and evaluating several models, with Gradient Boosting performing the best.

---

## 🛠️ Technologies Used

- **Python:** The core programming language.
- **Streamlit:** For building the interactive web application.
- **Pandas:** For data manipulation and analysis.
- **Scikit-learn:** For the machine learning model, imputation, and scaling.
- **Matplotlib:** For creating the comparison box plot.
- **Git & Git LFS:** For version control and managing large files.
