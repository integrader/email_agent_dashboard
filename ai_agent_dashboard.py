# Import required libraries
import streamlit as st
import pandas as pd
import requests
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.onnx import export, FeaturesManager
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Title and description
st.title("AI Agent Dashboard")
st.write("Upload your data, define a query, fetch web results, extract information using ONNX, and download results.")

# Section 1: Data Upload or Google Sheets Connection
st.header("Step 1: Upload Data")

# File upload option
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Google Sheets connection option
sheet_id = st.text_input("Alternatively, enter your Google Sheet ID:")

# Initialize a DataFrame
df = None

# Handle uploaded file
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("CSV file uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

if sheet_id:
    try:
        creds = service_account.Credentials.from_service_account_file(
            "path/to/your-credentials.json",  # Replace with your credentials file path
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
        )
        service = build("sheets", "v4", credentials=creds)
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=sheet_id, range="Sheet1").execute()
        values = result.get("values", [])
        
        if values:
            df = pd.DataFrame(values[1:], columns=values[0])
            st.success("Google Sheets data fetched successfully!")
        else:
            st.warning("No data found in the selected sheet.")
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")

# Proceed only if data is available
if df is not None:
    # Display data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Section 2: Column Selection and Dynamic Query Input
    st.header("Step 2: Define a Dynamic Query")
    column_options = df.columns.tolist()
    selected_column = st.selectbox("Select a column to work with:", column_options)

    if selected_column:
        st.write(f"Preview of '{selected_column}' column:")
        st.dataframe(df[[selected_column]].head())

        # Dynamic query input
        st.subheader("Define Your Query")
        st.write("Use `{entity}` as a placeholder for values in the selected column.")

        user_query = st.text_input(
            "Enter your query:",
            placeholder="E.g., 'Find the email address of {entity}'",
            key="query_input"
        )

        if user_query:
            if "{entity}" not in user_query:
                st.error("Your query must include the `{entity}` placeholder.")
            else:
                st.success("Your query is valid!")

                # Section 3: API Integration for Web Search
                st.header("Step 3: Fetch Web Search Results")
                api_key = st.text_input("Enter your SerpAPI key:", key="api_key_input")

                if api_key:
                    st.write("Fetching results...")

                    search_results = []
                    sample_values = df[selected_column].head(5).tolist()

                    for entity in sample_values:
                        query = user_query.replace("{entity}", entity)
                        url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}"

                        try:
                            response = requests.get(url)
                            if response.status_code == 200:
                                result = response.json()
                                search_results.append({
                                    "Entity": entity,
                                    "Query": query,
                                    "Top Result": result.get("organic_results", [{}])[0].get("link", "No results")
                                })
                            else:
                                st.error(f"Failed to fetch results for '{entity}': {response.status_code}")
                        except Exception as e:
                            st.error(f"Error fetching results for '{entity}': {e}")

                    if search_results:
                        st.subheader("Search Results")
                        results_df = pd.DataFrame(search_results)
                        st.dataframe(results_df)

                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="search_results.csv",
                            mime="text/csv"
                        )

                        # Section 4: LLM Integration Using ONNX
                        st.header("Step 4: Extract Information Using ONNX")

                        
                        def export_to_onnx(model_name, output_path):
                            # Load the tokenizer and model
                            tokenizer = AutoTokenizer.from_pretrained(model_name)
                            model = AutoModelForCausalLM.from_pretrained(model_name)

                            # Determine the correct feature and configuration for export
                            task = "causal-lm"  # Use "causal-lm" for text generation models
                            model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=task)
                            onnx_config = model_onnx_config(model.config)

                            # Ensure the output directory exists
                            output_dir = Path(output_path).parent
                            output_dir.mkdir(parents=True, exist_ok=True)

                            # Export the model to ONNX format
                            export(
                                preprocessor=tokenizer,
                                model=model,
                                config=onnx_config,
                                output=Path(output_path),
                                opset=14,  # Use ONNX opset 14
                            )
                            print(f"Model successfully exported to ONNX at {output_path}")

                        def load_onnx_model(onnx_path):
                            return ort.InferenceSession(onnx_path)

                        def process_with_onnx(session, tokenizer, prompts):
                            extracted_data = []
                            for prompt in prompts:
                                inputs = tokenizer(prompt, return_tensors="np", padding=True, truncation=True)
                                outputs = session.run(None, {"input_ids": inputs["input_ids"]})[0]
                                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                                extracted_data.append(result)
                            return extracted_data

                        onnx_path = "gpt_neo.onnx"
                        export_to_onnx("EleutherAI/gpt-neo-1.3B", onnx_path)
                        session = load_onnx_model(onnx_path)
                        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

                        prompts = [
                            f"Extract key information related to {row['Entity']} from the following URL: {row['Top Result']}. If no result is available, return 'No relevant information'."
                            for _, row in results_df.iterrows()
                        ]
                        extracted_results = process_with_onnx(session, tokenizer, prompts)

                        extracted_df = pd.DataFrame({"Entity": results_df["Entity"], "Extracted Information": extracted_results})
                        st.subheader("Extracted Information")
                        st.dataframe(extracted_df)

                        extracted_csv = extracted_df.to_csv(index=False)
                        st.download_button(
                            label="Download Extracted Data as CSV",
                            data=extracted_csv,
                            file_name="extracted_information.csv",
                            mime="text/csv"
                        )
