# Signature Identification System

A Python-based web application for signature identification, designed to identify the most likely registered writer using a trained machine-learning model.

## About the Project

The Signature Identification System provides a web interface for managing registered users and performing signature identification. It includes separate Student and Admin modules for user access and system management.

## Key Features

- **Signature Identification:** Analyze an uploaded signature and identify the closest registered match using the trained model.
- **Student Module:** Provides student-facing access to the application's available features.
- **Admin Module:** Provides administrative access to system management features.
- **Signature Management:** Organize registered signature samples.
- **Identification Model:** Uses a saved machine-learning model for identification.
- **Web Interface:** Built with Python, HTML, and CSS.

## Technology Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn / saved model (`identification.joblib`)
- **Frontend:** HTML, CSS
- **Backend:** Python web application
- **Tools:** Git, GitHub, VS Code

## Project Structure

```text
signature_system_upgrade/
├── main.py
├── desktop.py
├── create_admin.py
├── set_admin_role.py
├── models/
│   └── identification.joblib
├── static/
│   ├── style.css
│   └── signatures/
├── templates/
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/prathmeshgawande83-ship-it/signature-identification-system.git
cd signature-identification-system
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

Start the application using the entry point configured for your project. For a Flask application, this may be:

```bash
python main.py
```

Then open the local URL printed by the application in your browser.

## Important Notes

- Use authorized signature samples only.
- Do not commit real signatures, student records, passwords, or secret configuration values.
- Identification results should be treated as predictions, not definitive proof of identity.

## Author

**Prathmesh Gawande**  
BTech Computer Science & Engineering Student

GitHub: [@prathmeshgawande83-ship-it](https://github.com/prathmeshgawande83-ship-it)

---

*Developed as a Python project exploring signature identification and machine learning.*
