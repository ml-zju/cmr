import streamlit as st
import pandas as pd
from rdkit import Chem
from io import BytesIO
import pubchempy as pcp
from tools.get_cid import get_compound_cid
import asyncio
import aiohttp
import concurrent.futures
import time
from functools import lru_cache


def is_valid_smiles(smiles_string):
    try:
        molecule = Chem.MolFromSmiles(smiles_string, sanitize=False)
        return molecule is not None
    except:
        return False


@lru_cache(maxsize=1000)
def get_chemical_property_from_cid_cached(cid, property_name):
    try:
        compound = pcp.Compound.from_cid(cid)

        if property_name == 'smiles':
            return compound.canonical_smiles if compound.canonical_smiles else 'Not found'
        elif property_name == 'inchi':
            return compound.inchi if compound.inchi else 'Not found'
        elif property_name == 'inchikey':
            return compound.inchikey if compound.inchikey else 'Not found'
        elif property_name == 'molecular_formula':
            return compound.molecular_formula if compound.molecular_formula else 'Not found'
        elif property_name == 'iupac_name':
            return compound.iupac_name if compound.iupac_name else 'Not found'
        else:
            return 'Not found'
    except:
        return 'Not found'


@lru_cache(maxsize=1000)
def get_all_chemical_info_from_cid_cached(cid):
    try:
        compound = pcp.Compound.from_cid(cid)

        return {
            'smiles': compound.canonical_smiles if compound.canonical_smiles else 'Not found',
            'inchi': compound.inchi if compound.inchi else 'Not found',
            'inchikey': compound.inchikey if compound.inchikey else 'Not found',
            'molecular_formula': compound.molecular_formula if compound.molecular_formula else 'Not found',
            'iupac_name': compound.iupac_name if compound.iupac_name else 'Not found'
        }
    except:
        return {
            'smiles': 'Not found',
            'inchi': 'Not found',
            'inchikey': 'Not found',
            'molecular_formula': 'Not found',
            'iupac_name': 'Not found'
        }


def get_all_chemical_info_optimized(identifier):
    try:
        cid = get_compound_cid(identifier)
        if not cid:
            return {
                'CAS/Name': identifier,
                'SMILES': 'Not found',
                'InChI': 'Not found',
                'InChIKey': 'Not found',
                'Molecular Formula': 'Not found',
                'Chemical Name': 'Not found'
            }

        chemical_info = get_all_chemical_info_from_cid_cached(cid)

        return {
            'CAS/Name': identifier,
            'SMILES': chemical_info['smiles'],
            'InChI': chemical_info['inchi'],
            'InChIKey': chemical_info['inchikey'],
            'Molecular Formula': chemical_info['molecular_formula'],
            'Chemical Name': chemical_info['iupac_name']
        }

    except Exception as e:
        print(f"Error getting chemical info for {identifier}: {e}")
        return {
            'CAS/Name': identifier,
            'SMILES': 'Error',
            'InChI': 'Error',
            'InChIKey': 'Error',
            'Molecular Formula': 'Error',
            'Chemical Name': 'Error'
        }


async def get_chemical_info_async(session, identifier):
    try:
        cid = get_compound_cid(identifier)
        if not cid:
            return {
                'CAS/Name': identifier,
                'SMILES': 'Not found',
                'InChI': 'Not found',
                'InChIKey': 'Not found',
                'Molecular Formula': 'Not found',
                'Chemical Name': 'Not found'
            }

        base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"

        urls = {
            'smiles': f"{base_url}/{cid}/property/IsomericSMILES/JSON",
            'inchi': f"{base_url}/{cid}/property/InChI/JSON",
            'inchikey': f"{base_url}/{cid}/property/InChIKey/JSON",
            'molecular_formula': f"{base_url}/{cid}/property/MolecularFormula/JSON",
            'iupac_name': f"{base_url}/{cid}/property/IUPACName/JSON"
        }

        results = {}

        tasks = []
        for prop, url in urls.items():
            tasks.append(fetch_property(session, prop, url))

        property_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (prop, _) in enumerate(urls.items()):
            if isinstance(property_results[i], Exception):
                results[prop] = 'Error'
            else:
                results[prop] = property_results[i]

        return {
            'CAS/Name': identifier,
            'SMILES': results.get('smiles', 'Not found'),
            'InChI': results.get('inchi', 'Not found'),
            'InChIKey': results.get('inchikey', 'Not found'),
            'Molecular Formula': results.get('molecular_formula', 'Not found'),
            'Chemical Name': results.get('iupac_name', 'Not found')
        }

    except Exception as e:
        return {
            'CAS/Name': identifier,
            'SMILES': 'Error',
            'InChI': 'Error',
            'InChIKey': 'Error',
            'Molecular Formula': 'Error',
            'Chemical Name': 'Error'
        }


async def fetch_property(session, prop_name, url):
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                data = await response.json()
                properties = data.get('PropertyTable', {}).get('Properties', [])
                if properties:
                    prop_map = {
                        'smiles': 'IsomericSMILES',
                        'inchi': 'InChI',
                        'inchikey': 'InChIKey',
                        'molecular_formula': 'MolecularFormula',
                        'iupac_name': 'IUPACName'
                    }
                    return properties[0].get(prop_map[prop_name], 'Not found')
            return 'Not found'
    except:
        return 'Not found'


async def process_identifiers_async(identifiers):
    connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [get_chemical_info_async(session, identifier) for identifier in identifiers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'CAS/Name': identifiers[i],
                    'SMILES': 'Error',
                    'InChI': 'Error',
                    'InChIKey': 'Error',
                    'Molecular Formula': 'Error',
                    'Chemical Name': 'Error'
                })
            else:
                processed_results.append(result)

        return processed_results


def process_with_threading(identifiers, max_workers=5):
    def process_single(identifier):
        return get_all_chemical_info_optimized(identifier)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_identifier = {executor.submit(process_single, identifier): identifier
                                for identifier in identifiers}

        for future in concurrent.futures.as_completed(future_to_identifier):
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                identifier = future_to_identifier[future]
                results.append({
                    'CAS/Name': identifier,
                    'SMILES': 'Error',
                    'InChI': 'Error',
                    'InChIKey': 'Error',
                    'Molecular Formula': 'Error',
                    'Chemical Name': 'Error'
                })

    return results

def create_download_file(df, file_format):
    if file_format == "CSV":
        return df.to_csv(index=False).encode('utf-8')
    elif file_format == "Excel":
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Chemical_Info', index=False)
        return buffer.getvalue()


def smiles_conversion():
    def process_csv(file, processing_method) -> pd.DataFrame:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return f"Error reading the CSV file: {e}"

        if 'Identifier' not in df.columns:
            return "CSV file must contain a column named 'Identifier'."

        if not df['Identifier'].apply(lambda x: isinstance(x, str)).all():
            return "Each row in the 'Identifier' column should have a single string."

        identifier_list = [id.strip() for id in df['Identifier'] if pd.notna(id)]

        if processing_method == "Async (Fastest)":
            try:
                results = asyncio.run(process_identifiers_async(identifier_list))
            except Exception as e:
                st.error(f"Async processing failed: {e}. Falling back to threading method.")
                results = process_with_threading(identifier_list)
        elif processing_method == "Threading (Fast)":
            results = process_with_threading(identifier_list)
        else:  # Sequential
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, identifier in enumerate(identifier_list):
                status_text.text(f'Processing {i + 1}/{len(identifier_list)}: {identifier}')
                result = get_all_chemical_info_optimized(identifier)
                results.append(result)
                progress_bar.progress((i + 1) / len(identifier_list))

            progress_bar.empty()
            status_text.empty()

        return pd.DataFrame(results)

    st.title("Chemical Information Converter")
    st.write("Convert chemical identifiers (CAS numbers, names, SMILES) to various molecular formats")

    st.sidebar.header("Processing Options")
    processing_method = st.sidebar.selectbox(
        "Choose processing method:",
        ["Async (Fastest)", "Threading (Fast)", "Sequential (Slowest)"],
        help="Async: Uses asynchronous requests (fastest)\nThreading: Uses thread pool (fast)\nSequential: Processes one by one (slowest but most stable)"
    )

    if processing_method == "Threading (Fast)":
        max_workers = st.sidebar.slider("Max Workers", 1, 10, 5, help="Number of concurrent threads")

    option = st.selectbox("Choose an option:", ["Input CAS/Name", "Upload CSV File"])

    if option == "Input CAS/Name":
        st.subheader("Manual Input")

        identifiers = st.text_area(
            "Enter CAS/Name (one per line):",
            height=150,
            help="Enter multiple CAS numbers or names, one per line. Each entry should be a single string.",
            placeholder="Example:\n50-00-0\nCC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O"
        )

        col1, col2 = st.columns(2)
        with col1:
            output_format = st.selectbox("Choose output format:", ["CSV", "Excel"])
        with col2:
            molecule_format = st.selectbox("Choose molecule format to display:",
                                           ["SMILES", "InChI", "InChIKey", "Molecular Formula", "Chemical Name"])

        if st.button("Convert", type="primary"):
            if identifiers:
                identifier_list = [id.strip() for id in identifiers.split('\n') if id.strip()]

                if not identifier_list:
                    st.warning("Please enter valid identifiers.")
                    return

                start_time = time.time()

                if processing_method == "Async (Fastest)":
                    with st.spinner("Processing with async method..."):
                        try:
                            results = asyncio.run(process_identifiers_async(identifier_list))
                        except Exception as e:
                            st.error(f"Async processing failed: {e}. Falling back to threading method.")
                            results = process_with_threading(identifier_list)
                elif processing_method == "Threading (Fast)":
                    with st.spinner("Processing with threading method..."):
                        results = process_with_threading(identifier_list,
                                                         max_workers if 'max_workers' in locals() else 5)
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []

                    for i, identifier in enumerate(identifier_list):
                        status_text.text(f'Processing {i + 1}/{len(identifier_list)}: {identifier}')
                        chemical_info = get_all_chemical_info_optimized(identifier)
                        results.append(chemical_info)
                        progress_bar.progress((i + 1) / len(identifier_list))

                    progress_bar.empty()
                    status_text.empty()

                processing_time = time.time() - start_time

                full_results_df = pd.DataFrame(results)

                display_df = full_results_df[['CAS/Name', molecule_format]].copy()

                st.success(
                    f"Conversion completed in {processing_time:.2f} seconds! Processed {len(identifier_list)} identifiers.")
                st.write(f"**Converted {molecule_format}:**")
                st.dataframe(display_df, use_container_width=True)

                file_data = create_download_file(full_results_df, output_format)
                file_extension = "csv" if output_format == "CSV" else "xlsx"
                mime_type = "text/csv" if output_format == "CSV" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

                st.download_button(
                    label=f"📥 Download {output_format} (All Chemical Info)",
                    data=file_data,
                    file_name=f'converted_chemical_info.{file_extension}',
                    mime=mime_type
                )

                success_count = len([r for r in results if r['SMILES'] not in ['Not found', 'Error']])
                st.info(f"Successfully converted: {success_count}/{len(identifier_list)} identifiers")

            else:
                st.warning("Please enter CAS/Name(s) to convert, one per line.")

    elif option == "Upload CSV File":
        st.subheader("CSV File Upload")

        uploaded_file = st.file_uploader(
            "Upload your CSV file:",
            type="csv",
            help="CSV file must contain a column named 'Identifier' with chemical identifiers"
        )

        if uploaded_file:
            try:
                preview_df = pd.read_csv(uploaded_file)
                st.write("**File Preview:**")
                st.dataframe(preview_df.head(), use_container_width=True)

                if 'Identifier' not in preview_df.columns:
                    st.error("❌ CSV file must contain a column named 'Identifier'.")
                    return
                else:
                    st.success(f"✅ File loaded successfully! Found {len(preview_df)} identifiers.")

            except Exception as e:
                st.error(f"Error reading file: {e}")
                return

        col1, col2 = st.columns(2)
        with col1:
            output_format = st.selectbox("Choose output format:", ["CSV", "Excel"], key="csv_output")
        with col2:
            molecule_format = st.selectbox("Choose molecule format to display:",
                                           ["SMILES", "InChI", "InChIKey", "Molecular Formula", "Chemical Name"],
                                           key="csv_format")

        if st.button("Process CSV", type="primary"):
            if uploaded_file:
                start_time = time.time()

                uploaded_file.seek(0)

                if processing_method != "Sequential (Slowest)":
                    with st.spinner(f"Processing with {processing_method.lower()} method..."):
                        result = process_csv(uploaded_file, processing_method)
                else:
                    result = process_csv(uploaded_file, processing_method)

                processing_time = time.time() - start_time

                if isinstance(result, str):
                    st.error(result)
                else:
                    display_df = result[['CAS/Name', molecule_format]].copy()

                    st.success(
                        f"Processing completed in {processing_time:.2f} seconds! Processed {len(result)} identifiers.")
                    st.write(f"**Converted {molecule_format}:**")
                    st.dataframe(display_df, use_container_width=True)

                    file_data = create_download_file(result, output_format)
                    file_extension = "csv" if output_format == "CSV" else "xlsx"
                    mime_type = "text/csv" if output_format == "CSV" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

                    st.download_button(
                        label=f"📥 Download {output_format} (All Chemical Info)",
                        data=file_data,
                        file_name=f'converted_chemical_info.{file_extension}',
                        mime=mime_type
                    )

                    success_count = len(result[result['SMILES'].isin(['Not found', 'Error']) == False])
                    st.info(f"Successfully converted: {success_count}/{len(result)} identifiers")

            else:
                st.warning("Please upload a CSV file to process.")

    with st.expander("📖 Usage Instructions"):
        st.markdown("""
        ### How to use this tool:

        **Processing Methods:**
        - **Async (Fastest)**: Uses asynchronous HTTP requests for maximum speed
        - **Threading (Fast)**: Uses thread pool for concurrent processing
        - **Sequential (Slowest)**: Processes one identifier at a time (most stable)

        **Option 1: Manual Input**
        - Enter chemical identifiers (CAS numbers, chemical names, or SMILES) one per line
        - Select your preferred processing method, output format and molecule format
        - Click "Convert" to process

        **Option 2: CSV Upload**
        - Prepare a CSV file with a column named 'Identifier'
        - Upload the file and select your preferences
        - Click "Process CSV" to convert all identifiers

        ### Supported Input Formats:
        - CAS Registry Numbers (e.g., 50-00-0)
        - Chemical Names (e.g., water, caffeine)
        - SMILES strings (e.g., CCO)
        - InChI strings

        ### Output Formats:
        - **SMILES**: Simplified molecular-input line-entry system
        - **InChI**: International Chemical Identifier
        - **InChIKey**: Hashed version of InChI
        - **Molecular Formula**: Chemical formula (e.g., H2O)
        - **Chemical Name**: IUPAC name

        ### Performance Tips:
        - Use "Async" method for large datasets (>50 identifiers)
        - Use "Threading" method for medium datasets (10-50 identifiers)
        - Use "Sequential" method for small datasets or if other methods fail
        - Results are cached to speed up repeated queries

        ### Notes:
        - The tool queries PubChem database for chemical information
        - Async method is fastest but may be less stable with very large datasets
        - Download files contain all available chemical information
        """)


if __name__ == "__main__":
    smiles_conversion()