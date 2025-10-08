import streamlit as st
import pandas as pd
from llm.utils import get_llm

def data_annotation(model_name, temperature, api_key, secret_key, system, token, appid, api_secret, spark_url):
    st.title("🔖 Dataset Annotation Module")
    st.markdown(
        """
        **This module converts unstructured text into structured data using a large language model.**
        - Upload a CSV or Excel file containing at least one text column.
        - Choose which column to process.
        - Customize the prompt prefix.
        - You can resume incomplete annotations using the previously exported Excel file.
        """, unsafe_allow_html=True
    )

    st.markdown("**📄 Input file format example:**")
    st.code(
        """
id,text
1,"The shipment arrived late due to weather delays."
2,"Customer complained about billing errors in the invoice."
3,"System outage occurred at 3AM affecting server logs."
""", language='csv'
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file:",
        type=['csv', 'xlsx'],
        help="File must include at least one text column for annotation."
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        if not text_columns:
            st.error("No text columns found in the uploaded file for annotation.")
            return

        selected_column = st.selectbox(
            "Select the column to annotate:",
            options=text_columns,
            help="Choose which text column to feed into the LLM prompt."
        )

        annotation_column = st.text_input(
            "Name of the annotation column to create:",
            value="annotation",
            help="This new column will contain the model's structured output."
        )

        st.subheader("✏️ Prompt Prefix")
        st.markdown("Provide a prompt that will be prepended before each text row. The system will append the text to complete the prompt.")
        prompt_prefix = st.text_area(
            "Prompt Prefix:",
            value="Convert the following unstructured text into structured format, extracting key information:",
            height=100
        )

        resume_mode = annotation_column in df.columns
        start_index = df[annotation_column].notna().sum() if resume_mode else 0

        if st.button("🚀 Start Annotation"):
            st.info("Annotation in progress, please wait...")
            progress_bar = st.progress(start_index / len(df))

            llm = get_llm(
                model_name=model_name,
                temperature=temperature,
                api_key=api_key,
                secret_key=secret_key,
                system=system,
                token=token,
                appid=appid,
                api_secret=api_secret,
                spark_url=spark_url
            )

            if not resume_mode:
                df[annotation_column] = None

            total = len(df)
            for idx in range(start_index, total):
                row = df.iloc[idx]
                text_value = str(row[selected_column])
                prompt = f"{prompt_prefix}\n{text_value}"
                try:
                    response = llm.chat(prompt)
                except Exception:
                    response = llm(prompt)

                df.at[idx, annotation_column] = response
                st.markdown(f"**Row {idx + 1}/{total}:** {response}")
                progress_bar.progress((idx + 1) / total)

            st.success("✅ Annotation complete!")
            st.dataframe(df, use_container_width=True)

            excel = df.to_excel(index=False, engine='openpyxl')
            st.download_button(
                "📥 Download annotated data as Excel",
                data=excel,
                file_name="annotated_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("Please upload a CSV or Excel file to start annotation.")
