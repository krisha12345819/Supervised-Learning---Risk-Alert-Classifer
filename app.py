import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

st.set_page_config(page_title="Supervised Learning Project", layout="wide")

st.title("📊 Supervised Learning Project")
st.write("Upload dataset, train model, and make predictions.")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Information")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    columns = df.columns.tolist()

    target = st.selectbox("Select Target Column", columns)

    features = st.multiselect(
        "Select Feature Columns",
        [col for col in columns if col != target],
        default=[col for col in columns if col != target]
    )

    if len(features) > 0:

        X = df[features]
        y = df[target]

        # Handle categorical columns
        le_dict = {}

        for col in X.columns:
            if X[col].dtype == "object":
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le

        if y.dtype == "object":
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)

        task_type = st.radio(
            "Select Problem Type",
            ["Classification", "Regression"]
        )

        if task_type == "Classification":
            model_name = st.selectbox(
                "Choose Model",
                [
                    "Logistic Regression",
                    "Decision Tree",
                    "Random Forest"
                ]
            )
        else:
            model_name = st.selectbox(
                "Choose Model",
                [
                    "Linear Regression",
                    "Decision Tree",
                    "Random Forest"
                ]
            )

        test_size = st.slider(
            "Test Size",
            0.1,
            0.4,
            0.2
        )

        if st.button("Train Model"):

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42
            )

            # Classification Models
            if task_type == "Classification":

                if model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)

                elif model_name == "Decision Tree":
                    model = DecisionTreeClassifier(random_state=42)

                else:
                    model = RandomForestClassifier(
                        n_estimators=100,
                        random_state=42
                    )

            # Regression Models
            else:

                if model_name == "Linear Regression":
                    model = LinearRegression()

                elif model_name == "Decision Tree":
                    model = DecisionTreeRegressor(random_state=42)

                else:
                    model = RandomForestRegressor(
                        n_estimators=100,
                        random_state=42
                    )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            st.subheader("Model Performance")

            if task_type == "Classification":
                score = accuracy_score(y_test, predictions)
                st.success(f"Accuracy: {score:.4f}")

            else:
                score = r2_score(y_test, predictions)
                st.success(f"R² Score: {score:.4f}")

            st.subheader("Prediction Results")

            result_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": predictions
            })

            st.dataframe(result_df.head(20))

            csv = result_df.to_csv(index=False)

            st.download_button(
                "Download Predictions",
                csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

