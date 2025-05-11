# Data Validation and Anomaly Detection App

A Streamlit application for validating data and detecting anomalies using AI-powered algorithms.

## Features

- Multiple file format support (CSV, Excel, JSON, Parquet)
- AI-powered anomaly detection using Isolation Forest
- Automatic data correction
- Missing value handling
- Pattern detection
- Consistency checking
- Detailed validation reporting

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BulkUpload.git
cd BulkUpload
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Upload your data file and follow the on-screen instructions

## Project Structure

- `app.py`: Main Streamlit application
- `data_validator.py`: Data validation and anomaly detection logic
- `uploads/`: Directory for uploaded files
- `validated_data/`: Directory for validated and corrected data
- `models/`: Directory for saved models

## Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- joblib

## License

MIT License 