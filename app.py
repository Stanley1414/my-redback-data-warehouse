import streamlit as st
import os
from data_validator import DataValidator
import pandas as pd
from datetime import datetime
import json
import joblib
import io
import re

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
VALIDATED_FOLDER = 'validated_data'
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx', 'json', 'parquet'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VALIDATED_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_json_file(file_path):
    """Validate and convert JSON file to proper format."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Clean the content
        content = clean_json_string(content)
        
        # Try to parse as JSONL first (one object per line)
        try:
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            if all(line.startswith('{') and line.endswith('}') for line in lines):
                data = [json.loads(line) for line in lines]
                return pd.DataFrame(data)
        except json.JSONDecodeError:
            pass  # Not JSONL format, continue with other formats
        
        # Try to parse as standard JSON
        try:
            data = json.loads(content)
            
            # Handle array of objects
            if isinstance(data, list):
                if len(data) == 0:
                    raise ValueError("Empty JSON array")
                return pd.DataFrame(data)
            
            # Handle single object
            elif isinstance(data, dict):
                # Try to find a list of records in the dictionary
                for key, value in data.items():
                    if isinstance(value, list):
                        return pd.DataFrame(value)
                # If no list found, convert the single dictionary
                return pd.DataFrame([data])
            else:
                raise ValueError(f"Unsupported JSON structure: {type(data)}")
                
        except json.JSONDecodeError as e:
            error_msg = str(e)
            if "Extra data" in error_msg:
                # Try to fix common formatting issues
                try:
                    # Try to parse as array of objects
                    fixed_content = "[" + content.replace("}\n{", "},{") + "]"
                    data = json.loads(fixed_content)
                    return pd.DataFrame(data)
                except json.JSONDecodeError:
                    raise ValueError(
                        "Your JSON file contains multiple objects that need to be properly formatted.\n\n"
                        "Please format your JSON file in one of these ways:\n\n"
                        "1. As an array of objects:\n"
                        "[\n"
                        "    {\"key1\": \"value1\"},\n"
                        "    {\"key2\": \"value2\"}\n"
                        "]\n\n"
                        "2. As JSONL (one object per line):\n"
                        "{\"key1\": \"value1\"}\n"
                        "{\"key2\": \"value2\"}\n\n"
                        f"Error details: {error_msg}"
                    )
            elif "Expecting property name" in error_msg:
                raise ValueError(
                    "Invalid JSON format: Missing property name or trailing comma. "
                    f"Error details: {error_msg}"
                )
            else:
                raise ValueError(f"Invalid JSON format: {error_msg}")
                
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    except PermissionError:
        raise ValueError(f"Permission denied: {file_path}")
    except Exception as e:
        raise ValueError(f"Error processing JSON file: {str(e)}")

def clean_json_string(content: str) -> str:
    """Clean and normalize JSON string content."""
    # Remove BOM if present
    content = content.lstrip('\ufeff')
    # Remove comments (both single-line and multi-line)
    content = re.sub(r'//.*?$|/\*.*?\*/', '', content, flags=re.MULTILINE)
    # Remove trailing commas
    content = re.sub(r',(\s*[}\]])', r'\1', content)
    return content.strip()

def main():
    st.set_page_config(
        page_title="Data Validation App",
        page_icon="✅",
        layout="wide"
    )
    
    st.title("Data Validation and Anomaly Detection")
    st.write("Upload your data file to validate and detect anomalies.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=ALLOWED_EXTENSIONS,
        help=f"Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
    )
    
    if uploaded_file is not None:
        try:
            # Create timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = uploaded_file.name
            file_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{filename}")
            
            # Save uploaded file
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Handle JSON files specially
            if filename.lower().endswith('.json'):
                try:
                    # Convert JSON to DataFrame
                    df = validate_json_file(file_path)
                    # Save as CSV for consistent processing
                    csv_path = file_path.replace('.json', '.csv')
                    df.to_csv(csv_path, index=False)
                    file_path = csv_path
                except Exception as e:
                    st.error(f'Error processing JSON file: {str(e)}')
                    return
            
            # Initialize validator
            validator = DataValidator(model_name='sports_data')
            
            # Load data and train model
            data = validator.load_data(file_path)
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            # Show data info
            st.subheader("Data Information")
            buffer = io.StringIO()
            data.info(buf=buffer)
            st.text(buffer.getvalue())
            
            # Validate the uploaded data
            with st.spinner('Validating data and training model...'):
                corrected_data, report = validator.validate_and_correct(file_path, model_dir="models")
            
            # Display validation results
            st.subheader("Validation Results")
            
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", report['total_records'])
            with col2:
                st.metric("Anomalies Detected", report['anomalies_detected'])
            with col3:
                st.metric("Model Trained", "Yes" if report['model_trained'] else "No")
            
            # Display detailed results in expandable sections
            with st.expander("Missing Values Report"):
                if report['missing_values']:
                    missing_df = pd.DataFrame([
                        {'Column': k, 'Count': v['count'], 'Percentage': f"{v['percentage']:.2f}%"}
                        for k, v in report['missing_values'].items()
                    ])
                    st.dataframe(missing_df)
                else:
                    st.success("No missing values found!")
            
            with st.expander("Range Violations"):
                if report['range_violations']:
                    range_df = pd.DataFrame([
                        {'Column': k, 'Violations': v['count']}
                        for k, v in report['range_violations'].items()
                    ])
                    st.dataframe(range_df)
                else:
                    st.success("No range violations found!")
            
            with st.expander("Pattern Violations"):
                if report['pattern_violations']:
                    pattern_df = pd.DataFrame([
                        {'Pattern': k, 'Violations': v['count']}
                        for k, v in report['pattern_violations'].items()
                    ])
                    st.dataframe(pattern_df)
                else:
                    st.success("No pattern violations found!")
            
            # Save corrected data
            output_filename = f"{timestamp}_corrected_{filename}"
            output_path = os.path.join(VALIDATED_FOLDER, output_filename)
            corrected_data.to_csv(output_path, index=False)
            
            # Download button for corrected data
            st.download_button(
                label="Download Corrected Data",
                data=corrected_data.to_csv(index=False).encode('utf-8'),
                file_name=output_filename,
                mime='text/csv'
            )
            
        except Exception as e:
            st.error(f'Error processing file: {str(e)}')

if __name__ == '__main__':
    main() 