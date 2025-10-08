from online.qa_interface import qa_interface
from online.CMR_prediction import molecular_prediction
from online.smiles_conversion import smiles_conversion
from online.report_generation import generate_report_main
from online.data_annotation import data_annotation
import sys
import os
from llm.model_name import MODEL_CATEGORIES, SPARK_MODELS
from PIL import Image
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.markdown("""
    <style>
        .block-container {
            max-width: 1000px;
            padding: 2rem;
            margin: 0 auto;
        }
    </style>
""", unsafe_allow_html=True)

def show_model_config():
    """Displays the sidebar configuration for model settings."""
    st.sidebar.header("Model Parameters Configuration")

    model_category = st.sidebar.selectbox("Select Model Category:", list(MODEL_CATEGORIES.keys()))
    model_name = st.sidebar.selectbox("Select Model:", MODEL_CATEGORIES[model_category])
    api_key = st.sidebar.text_input("Enter your API Key:", type="password")

    appid = api_secret = spark_url = secret_key = system = token = None

    if model_category == "Spark":
        appid = st.sidebar.text_input("Enter App ID:")
        api_secret = st.sidebar.text_input("Enter API Secret:")
        spark_url = SPARK_MODELS.get(model_name.lower(), ('', ''))[0]
    elif model_category == "ERNIE":
        secret_key = st.sidebar.text_input("Enter Secret Key:")

    temperature = st.sidebar.slider("Select Temperature:", min_value=0.0, max_value=1.0, value=0.1)

    if st.sidebar.button("Save Settings"):
        st.session_state.model_config = {
            "model_name": model_name,
            "temperature": temperature,
            "api_key": api_key,
            "secret_key": secret_key,
            "system": system,
            "token": token,
            "appid": appid,
            "api_secret": api_secret,
            "spark_url": spark_url
        }
        st.session_state.page = st.session_state.selected_page


def main():
    """Main function to initialize the sidebar navigation and display content based on user selection."""
    st.sidebar.title("Navigation Panel")

    # Use custom HTML and CSS for better button layout and spacing
    st.sidebar.markdown("""
        <style>
            .sidebar-button {
                width: 100%;
                padding: 10px;
                margin: 5px 0;
                text-align: left;
                display: block;  
            }
            .stButton button {
                text-align: left !important;
                justify-content: flex-start !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Create functional navigation panel with icons and aligned buttons
    if st.sidebar.button("📘 Application Manual", key="Instructions", help="Instructions for using the app",
                         use_container_width=True):
        st.session_state.page = "Instructions"

    if st.sidebar.button("💻 CMR Property Prediction", key="Prediction", help="Predict molecular CMR properties",
                         use_container_width=True):
        st.session_state.page = "Molecular Property Prediction"

    if st.sidebar.button("💬 Compound Property Q&A", key="QA", help="Ask questions about substance properties",
                         use_container_width=True):
        st.session_state.page = "Substance Property Q&A"

    if st.sidebar.button("📊 Prediction Report", key="Report", help="View prediction analysis report",
                         use_container_width=True):
        st.session_state.page = "Prediction Analysis Report"

    if st.sidebar.button("🧪 Structure Conversion", key="Tool", help="Convert SMILES structure",
                         use_container_width=True):
        st.session_state.page = "SMILES Conversion Toolkit"
    if st.sidebar.button("✏️ Data Annotation", key="Annotate", help="Convert unstructured text into a structured format",
                         use_container_width=True):
        st.session_state.page = "Data Annotation"

    # Set default page to "Instructions" if no page is selected
    if 'page' not in st.session_state:
        st.session_state.page = "Instructions"

    st.session_state.selected_page = st.session_state.page

    # Show model configuration settings in the sidebar
    show_model_config()

    # Display the appropriate page based on user selection
    if st.session_state.page == "Substance Property Q&A":
        if 'model_config' in st.session_state:
            config = st.session_state.model_config
            st.success(f"Loading {config['model_name']} for Q&A...")
            qa_interface(
                model_name=config['model_name'],
                temperature=config['temperature'],
                api_key=config['api_key'],
                secret_key=config['secret_key'],
                system=config['system'],
                token=config['token'],
                appid=config['appid'],
                api_secret=config['api_secret'],
                spark_url=config['spark_url']
            )
        else:
            st.warning("Please configure the model first.")

    elif st.session_state.page == "Molecular Property Prediction":
        st.header("CMR Property Prediction")
        if 'model_config' in st.session_state:
            config = st.session_state.model_config
            st.success(f"Loading {config['model_name']} for Molecular Prediction...")
            molecular_prediction(
                model_name=config['model_name'],
                temperature=config['temperature'],
                api_key=config['api_key'],
                secret_key=config['secret_key'],
                system=config['system'],
                token=config['token'],
                appid=config['appid'],
                api_secret=config['api_secret'],
                spark_url=config['spark_url']
            )
        else:
            st.warning("Please configure the model first.")

    elif st.session_state.page == "SMILES Conversion Toolkit":
        st.header("CAS/Name to SMILES Converter")
        smiles_conversion()

    elif st.session_state.page == "Prediction Analysis Report":
        st.header("CMR Prediction Report")
        if 'model_config' in st.session_state:
            config = st.session_state.model_config
            st.success(f"Generating report for {config['model_name']}...")
            generate_report_main(
                model_name=config['model_name'],
                temperature=config['temperature'],
                api_key=config['api_key'],
                secret_key=config['secret_key'],
                system=config['system'],
                token=config['token'],
                appid=config['appid'],
                api_secret=config['api_secret'],
                spark_url=config['spark_url']
            )
        else:
            st.warning("Please configure the model first.")

    elif st.session_state.page == "Data Annotation":
        st.header("Dataset Annotation")
        if 'model_config' in st.session_state:
            config = st.session_state.model_config
            data_annotation(
                model_name=config['model_name'],
                temperature=config['temperature'],
                api_key=config['api_key'],
                secret_key=config['secret_key'],
                system=config['system'],
                token=config['token'],
                appid=config['appid'],
                api_secret=config['api_secret'],
                spark_url=config['spark_url']
            )
        else:
            st.warning("Please configure the model first.")

    elif st.session_state.page == "Instructions":

        st.header("CMRAgent")

        st.write("""
        **CMRAgent** is a vertical LLM-based agent designed to transform chemical safety assessment from labor-intensive, compliance-driven workflows into an **automated, intelligent, and evidence-based** process.  
        It integrates function-calling LLMs, the ReAct (Reasoning–Acting) paradigm, over **100 custom retrieval tools** (via LangChain), and **CMRScreen**—a semi-supervised DMPNN model—for high-accuracy prediction of carcinogenic (C), mutagenic (M), and reproductive-toxic (R) endpoints.  
        In systematic benchmarks, CMRAgent achieved **state-of-the-art** accuracy across three CMR endpoints (**89.03%**, **85.35%**, **89.53%**) and >85% accuracy in chemical safety Q&A, significantly outperforming general-purpose LLMs (<35%).  
        By reducing knowledge gaps, mitigating hallucinations, and overcoming reliance on animal testing, CMRAgent lays the groundwork for a **unified, scalable safety evaluation ecosystem**.
        """)

        image = Image.open("images/CMRAgent.tif")
        st.image(image, caption="CMRAgent: AI-driven workflow for chemical safety")

        st.write("""
        ### Core Functionalities
        1. **Chemical Q&A**  
           - Molecular name conversion, physicochemical properties, ecology, human toxicity, safety & hazards.
        2. **CMR Property Prediction**  
           - Single or batch predictions with labels, confidence scores, applicability domain checks.
        3. **Regulatory-Compliant Report Generation**  
           - Auto-generated reports with compound data, prediction results, and supporting experimental info.

        ---

        ### Getting Started
        1. **Select Functionality** from the sidebar.  
        2. **Configure Model**: choose (e.g., DeepSeek, ChatGLM, Gemini), set parameters & API key.  
           - [DeepSeek](https://platform.deepseek.com/usage)  
           - [Gemini](https://aistudio.google.com/app/apikey)  
           - [ChatGLM](https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys)  
           - [Spark](https://xinghuo.xfyun.cn/sparkapi)  
           - [ERNIE](https://bailian.console.aliyun.com/?apiKey=1#/api-key)
        3. **Input Data**:  
           - For CMR Property Prediction: upload CSV with `SMILES` column.  
           - For Q&A: enter natural language queries.
        4. **Run & Review** results in the output panel; download reports if needed.

        **Note:** Ensure valid model configuration and API key before running tasks.
        """)

if __name__ == "__main__":
    main()
