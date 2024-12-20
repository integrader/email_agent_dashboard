# AI Agent Dashboard with ONNX Integration

## Project Summary
The **AI Agent Dashboard** is a versatile tool that allows users to dynamically process datasets, perform web searches, extract key information using ONNX-powered large language models (LLMs), and download the results for further analysis. This tool integrates **Hugging Face models**, **ONNX Runtime**, and **third-party APIs** like SerpAPI to streamline information retrieval and text processing.

### Key Features
- **Dynamic Dataset Processing**: Upload CSV files or fetch data from Google Sheets.
- **Custom Query Definition**: Define dynamic queries with placeholders for dataset entries.
- **Web Search Integration**: Use SerpAPI to fetch top search results for the queries.
- **Information Extraction with ONNX**: Leverage Hugging Face models converted to ONNX for efficient inference.
- **Interactive UI**: Built with Streamlit for a user-friendly and visually rich experience.

---

## Setup Instructions

### Prerequisites
- **Python**: Version 3.8 or higher.
- **Google Cloud Service Account**: Required for accessing Google Sheets (optional).
- **SerpAPI Key**: Required for performing web searches.

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/integrader/email_ai_dashboard.git
   cd ai-agent-dashboard


Install Dependencies Install the required Python libraries:
pip install -r requirements.txt


# Run the Application Start the Streamlit application:

streamlit run ai_agent_dashboard.py

# Usage Guide
## Workflow Steps
Upload Data

## Upload a CSV file or fetch data from Google Sheets.
Define Query

## Select a column and define a query using {entity} as a placeholder.
Perform Web Search

## Enter your SerpAPI key to fetch top web search results.
Extract Information

## Hugging Face models are exported to ONNX and used to extract key information from the web search results.
Download Results

## View and download the extracted information as a CSV file.


# Third-Party APIs and Tools
1. SerpAPI
Provides web search capabilities for queries.

2. Hugging Face Transformers
Pre-trained language models for natural language processing tasks.
Documentation
3. ONNX and ONNX Runtime
Converts Hugging Face models into an optimized format for efficient inference.
ONNX | ONNX Runtime
4. Google Sheets API
Enables fetching data dynamically from Google Sheets.

5. Streamlit
Framework for building interactive dashboards.


# Install these dependencies using:


pip install -r requirements.txt

# Contributing
Feel free to fork this repository, submit issues, or create pull requests to improve the tool. Contributions are always welcome!


---

### This project is licensed under the MIT License. See the LICENSE file for details.

yaml



### **Markdown Preview Tips**
- In VS Code, use **Markdown Preview** to see the rendered document:
  - Right-click inside the editor and select **Open Preview**.
  - Shortcut: Press `Ctrl+Shift+V` (Windows/Linux) or `Cmd+Shift+V` (macOS).

