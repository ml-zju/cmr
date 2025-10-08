import streamlit as st
import pandas as pd
from agents.agent import CMR
from tools import load_llm_tools
from llm.utils import get_llm
from rdkit import Chem
import time
import io


def validate_smiles(smiles_list):
    if not smiles_list:
        return False, "No SMILES strings provided."

    invalid_smiles = []
    for smiles in smiles_list:
        if not Chem.MolFromSmiles(smiles):
            invalid_smiles.append(smiles)

    if invalid_smiles:
        return False, f"Invalid SMILES strings found: {', '.join(invalid_smiles)}"
    return True, ""


def read_uploaded_file(uploaded_file):
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension == 'csv':
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='gbk')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin-1')

        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file, engine='openpyxl' if file_extension == 'xlsx' else 'xlrd')

        elif file_extension == 'txt':
            try:
                uploaded_file.seek(0)
                content = uploaded_file.read().decode('utf-8')
            except UnicodeDecodeError:
                try:
                    uploaded_file.seek(0)
                    content = uploaded_file.read().decode('gbk')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    content = uploaded_file.read().decode('latin-1')

            lines = content.strip().split('\n')
            if not lines:
                raise ValueError("Empty file")

            first_line = lines[0]

            if '\t' in first_line:
                df = pd.read_csv(io.StringIO(content), sep='\t')
            elif ',' in first_line:
                df = pd.read_csv(io.StringIO(content), sep=',')
            elif ';' in first_line:
                df = pd.read_csv(io.StringIO(content), sep=';')
            else:
                smiles_list = [line.strip() for line in lines if line.strip()]
                df = pd.DataFrame({'SMILES': smiles_list})

        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        return df, None

    except Exception as e:
        return None, f"Error reading file: {str(e)}"


def detect_smiles_column(df):
    possible_smiles_columns = [
        'SMILES', 'smiles', 'Smiles',
        'SMILE', 'smile', 'Smile',
        'Structure', 'structure', 'STRUCTURE',
        'Molecular_Structure', 'molecular_structure',
        'Chemical_Structure', 'chemical_structure'
    ]

    for col in possible_smiles_columns:
        if col in df.columns:
            return col

    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['smiles', 'smile', 'structure']):
            return col

    if len(df.columns) > 0:
        return df.columns[0]

    return None


def preview_file_data(df, smiles_column):
    st.write("**File preview:**")
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.write(f"Detected SMILES column: **{smiles_column}**")

    st.write("First 5 rows:")
    st.dataframe(df.head())

    if smiles_column in df.columns:
        smiles_data = df[smiles_column].dropna()
        st.write(f"Valid SMILES entries: {len(smiles_data)}")
        st.write(f"Missing values: {df[smiles_column].isna().sum()}")

        if len(smiles_data) > 0:
            st.write("SMILES examples:")
            examples = smiles_data.head(3).tolist()
            for i, smiles in enumerate(examples, 1):
                st.write(f"  {i}. {smiles}")


def format_results_display(results_df, category):
    category_mapping = {
        'Carcinogenic': 'C',
        'Mutagenic': 'M',
        'Toxic to Reproduction': 'R'
    }

    category_short = category_mapping.get(category, category)

    required_columns = [
        'smiles',
        f'label_{category_short}',
        f'calibrated_confid_{category_short}',
        f'in_domain_{category_short}'
    ]

    for col in required_columns:
        if col not in results_df.columns:
            raise KeyError(
                f"Required column '{col}' not found in results dataframe. Available columns: {list(results_df.columns)}")

    display_df = results_df[[
        'smiles',
        f'label_{category_short}',
        f'calibrated_confid_{category_short}',
        f'in_domain_{category_short}'
    ]].copy()

    display_df[f'label_{category_short}'] = display_df[f'label_{category_short}'].map({
        1: 'Positive',
        0: 'Negative'
    })

    display_df = display_df.round(3)

    display_df.columns = [
        'SMILES',
        'Prediction',
        'Confidence',
        'In Domain'
    ]
    return display_df


def perform_prediction(model, smiles, category):
    start_time = time.time()
    results_df = None

    category_mapping = {
        'Carcinogenic': 'C',
        'Mutagenic': 'M',
        'Toxic to Reproduction': 'R'
    }

    property_type = category_mapping.get(category, category)

    cmr_tool = next((tool for tool in model.tools if tool.name == "CMR Prediction"), None)

    if cmr_tool:
        try:
            tool_input = str({
                'smiles': [smiles],
                'property_type': property_type
            })
            config = {"callbacks": None}
            results_df = cmr_tool.run(tool_input, config=config)

            if not isinstance(results_df, pd.DataFrame):
                st.error(f"Expected a DataFrame but got: {type(results_df)}")
                return None, 0

            category_short = property_type
            results_df = results_df.rename(columns={
                'label': f'label_{category_short}',
                'calibrated_confid': f'calibrated_confid_{category_short}',
                'in_domain': f'in_domain_{category_short}'
            })

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.error(f"Error type: {type(e).__name__}")
            import traceback
            st.error(f"Detailed error traceback:\n{traceback.format_exc()}")
            return None, 0

    prediction_time = time.time() - start_time
    return results_df, prediction_time


def get_llm_response(model, smiles_list, category, results_df):
    try:
        category_mapping = {
            'Carcinogenic': 'C',
            'Mutagenic': 'M',
            'Toxic to Reproduction': 'R'
        }

        category_short = category_mapping.get(category, category)

        predictions_summary = []
        for i, row in results_df.iterrows():
            label_col = f'label_{category_short}'
            confid_col = f'calibrated_confid_{category_short}'

            if label_col in row and confid_col in row:
                predictions_summary.append(
                    f"Molecule {i + 1} (SMILES: {row['smiles']}): "
                    f"Prediction: {'Positive' if row[label_col] == 1 else 'Negative'}, "
                    f"Confidence: {row[confid_col]:.3f}"
                )

        prompt = (
            f"Analysis of {category} properties:\n"
            f"{chr(10).join(predictions_summary)}\n"
            "Please provide a concise summary of these predictions."
        )

        response = model.llm(prompt)
        return response
    except Exception as e:
        st.error(f"Error getting LLM response: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return "Unable to generate summary response."


def clean_dataframe_columns(df):
    columns_to_drop = [col for col in df.columns if col.endswith('_x') or col.endswith('_y')]
    error_columns = [col for col in df.columns if 'error' in col.lower()]
    columns_to_drop.extend(error_columns)

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    return df


def combine_category_results(all_results, category_results):
    if all_results.empty:
        return category_results

    common_columns = list(set(all_results.columns) & set(category_results.columns))

    if common_columns == ['smiles']:
        return pd.merge(all_results, category_results, on='smiles', how='outer')

    expected_common = ['smiles']
    unexpected_common = [col for col in common_columns if col not in expected_common]

    if unexpected_common:
        rename_dict = {}
        for col in unexpected_common:
            if col != 'smiles':
                new_name = f"{col}_temp"
                rename_dict[col] = new_name

        if rename_dict:
            category_results = category_results.rename(columns=rename_dict)

    merged_df = pd.merge(all_results, category_results, on='smiles', how='outer')

    return merged_df


def display_results_without_download(all_results, performance_stats, selected_categories, total_time, smiles_list):
    st.success(f"Total prediction time: {total_time:.2f} seconds")

    perf_df = pd.DataFrame(performance_stats)
    avg_time_per_mol = perf_df.groupby('molecule')['prediction_time'].mean()

    if not all_results.empty:
        all_results = clean_dataframe_columns(all_results)

        st.write("Prediction Results:")
        st.write(f"Results shape: {all_results.shape}")

        for category in selected_categories:
            st.write(f"\n{category} Predictions:")
            try:
                formatted_results = format_results_display(all_results, category)
                st.dataframe(formatted_results)
            except KeyError as e:
                st.error(f"Error formatting results for {category}: {str(e)}")
                continue

        for category in selected_categories:
            try:
                response = get_llm_response(st.session_state.selected_model,
                                            smiles_list,
                                            category,
                                            all_results)
                st.write(f"\nModel response for {category}:", response)
            except Exception as e:
                st.error(f"Error generating LLM response for {category}: {str(e)}")

    st.write(f"Average prediction time per molecule: {avg_time_per_mol.mean():.3f} seconds")
    return all_results


def molecular_prediction(model_name, temperature, api_key, secret_key=None, system=None, token=None, appid=None,
                         api_secret=None, spark_url=None):
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'results_ready' not in st.session_state:
        st.session_state.results_ready = False

    if 'selected_model' not in st.session_state:
        try:
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
            tools = load_llm_tools(llm, verbose=True)

            if not any(tool.name == "CMR Prediction" for tool in tools):
                st.error("CMR Prediction tool failed to load.")
                return

            st.session_state.selected_model = CMR(llm=llm, tools=tools)
            st.success(f"Model '{model_name}' and tools initialized successfully.")
        except Exception as e:
            st.error(f"Error initializing model and tools: {str(e)}")
            return

    input_method = st.selectbox("Select input method:", ["Manual Input", "Upload File"])

    st.write("Select properties to predict:")
    col1, col2, col3 = st.columns(3)
    with col1:
        carcinogenic = st.checkbox("Carcinogenic")
    with col2:
        mutagenic = st.checkbox("Mutagenic")
    with col3:
        reproductive = st.checkbox("Toxic to Reproduction")

    selected_categories = []
    if carcinogenic:
        selected_categories.append("Carcinogenic")
    if mutagenic:
        selected_categories.append("Mutagenic")
    if reproductive:
        selected_categories.append("Toxic to Reproduction")

    if not selected_categories:
        st.warning("Please select at least one property to predict.")
        return

    if input_method == "Manual Input":
        with st.form(key='molecular_prediction_form', clear_on_submit=True):
            smiles = st.text_area("Enter SMILES string (one per line):", height=80, key="prediction_smiles")
            submit_button = st.form_submit_button("Predict")

        if submit_button and smiles:
            process_predictions(smiles, selected_categories)

    elif input_method == "Upload File":
        st.markdown("### Upload File")
        st.markdown("""
        **Supported File Formats:**
        - **CSV files** (.csv): Comma-separated values
        - **Excel files** (.xlsx, .xls): Microsoft Excel format
        - **Text files** (.txt): Tab-separated, comma-separated, or one SMILES per line

        **File Format Requirements:**
        - The file should contain a column with SMILES strings
        - Common column names: SMILES, smiles, Structure, etc.
        - If no header is detected, the first column will be used as SMILES
        - For conversion of CAS registry numbers or IUPAC names to SMILES notation, please utilize the **SMILES Conversion Tool**.
        """)

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "xls", "txt"],
            help="Upload CSV, Excel, or TXT file containing SMILES strings"
        )

        if uploaded_file is not None:
            df, error_message = read_uploaded_file(uploaded_file)

            if error_message:
                st.error(error_message)
                return

            if df is None or df.empty:
                st.error("The uploaded file is empty or could not be read.")
                return

            smiles_column = detect_smiles_column(df)

            if smiles_column is None:
                st.error("Could not detect SMILES column. Please ensure your file contains a column with SMILES data.")
                return

            st.write("**Column Selection:**")
            available_columns = df.columns.tolist()
            selected_smiles_column = st.selectbox(
                "Select the column containing SMILES:",
                available_columns,
                index=available_columns.index(smiles_column) if smiles_column in available_columns else 0
            )

            preview_file_data(df, selected_smiles_column)

            predict_button = st.button("Predict", key="file_predict_button")

            if predict_button:
                if selected_smiles_column not in df.columns:
                    st.error(f"Selected column '{selected_smiles_column}' not found in the file.")
                    return

                smiles_data = df[selected_smiles_column].dropna().tolist()

                if not smiles_data:
                    st.error("No valid SMILES data found in the selected column.")
                    return

                smiles_string = "\n".join([str(s).strip() for s in smiles_data if str(s).strip()])

                if not smiles_string:
                    st.error("No valid SMILES strings found after processing.")
                    return

                st.info(f"Processing {len(smiles_data)} SMILES from uploaded file...")
                process_predictions(smiles_string, selected_categories)

    if st.session_state.results_ready and st.session_state.prediction_results is not None:
        st.markdown("---")
        st.markdown("### Download Results")

        clean_results = clean_dataframe_columns(st.session_state.prediction_results.copy())

        csv = clean_results.to_csv(index=False)
        st.download_button(
            label="📥 Download prediction results as CSV",
            data=csv,
            file_name='prediction_results.csv',
            mime='text/csv',
        )


def process_predictions(smiles, selected_categories):
    total_start_time = time.time()
    smiles_list = [s.strip() for s in smiles.split('\n') if s.strip()]
    is_valid, error_message = validate_smiles(smiles_list)

    if not is_valid:
        st.error(error_message)
        return

    if not st.session_state.selected_model:
        st.error("Model not initialized.")
        return

    try:
        progress_bar = st.progress(0)
        progress_text = st.empty()

        performance_stats = {
            'molecule': [],
            'category': [],
            'prediction_time': [],
            'in_domain': []
        }

        all_results = pd.DataFrame()
        total_molecules = len(smiles_list)
        total_categories = len(selected_categories)
        total_predictions = total_molecules * total_categories

        prediction_counter = 0

        category_mapping = {
            'Carcinogenic': 'C',
            'Mutagenic': 'M',
            'Toxic to Reproduction': 'R'
        }

        for category in selected_categories:
            st.write(f"\nProcessing {category} predictions...")
            category_results = pd.DataFrame()
            category_short = category_mapping.get(category, category)

            for idx, smiles in enumerate(smiles_list):
                molecule_name = f"Molecule {idx + 1}"
                progress_text.text(f"Processing molecule {idx + 1}/{len(smiles_list)} for {category}")

                results_df, pred_time = perform_prediction(
                    st.session_state.selected_model,
                    smiles,
                    category
                )

                performance_stats['molecule'].append(molecule_name)
                performance_stats['category'].append(category)
                performance_stats['prediction_time'].append(pred_time)

                if results_df is not None and not results_df.empty:
                    in_domain_col = f'in_domain_{category_short}'
                    performance_stats['in_domain'].append(
                        results_df[in_domain_col].iloc[0] if in_domain_col in results_df.columns else False
                    )
                else:
                    performance_stats['in_domain'].append(False)

                if results_df is not None:
                    if category_results.empty:
                        category_results = results_df
                    else:
                        category_results = pd.concat([category_results, results_df], ignore_index=True)

                prediction_counter += 1
                progress_bar.progress(prediction_counter / total_predictions)

            if not category_results.empty:
                all_results = combine_category_results(all_results, category_results)

        total_time = time.time() - total_start_time

        final_results = display_results_without_download(all_results, performance_stats, selected_categories,
                                                         total_time, smiles_list)

        st.session_state.prediction_results = final_results
        st.session_state.results_ready = True

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.error(f"Detailed error traceback:\n{traceback.format_exc()}")