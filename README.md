# 📊 Career Point Report Generator

![Logo](templates/assets/wordLogo.png)

A comprehensive and intuitive application for generating detailed student performance reports for Career Point coaching center. This tool helps educators and administrators to analyze student performance in various exams like NEET and JEE, both at a class-level and individual student level.

---

## ✨ Features

- **Multiple Exam Types:** Supports both NEET and JEE exam formats.
- **Flexible Report Generation:**
    - **Unit Test Reports:** Generate individual student reports and consolidated class reports.
    - **Chapter-wise Reports:** Create detailed subject-based reports for in-depth analysis.
- **Interactive UI:** Built with Streamlit for an easy-to-use and interactive user experience.
- **Rich Data Visualization:** Generates various plots and charts to visualize student and class performance, including:
    - Score Distribution Charts
    - Pass/Fail Donut Charts
    - Subject Accuracy Radar Charts
    - Student vs. Class Average Comparison
    - Individual Student Progress Tracking
- **PDF Report Generation:** Creates professional and well-formatted PDF reports that are easy to share and print.
- **Insightful Analytics:** Provides actionable insights and performance summaries to help identify student strengths and weaknesses.

---

## 🚀 How to Use

### **1. Prerequisites**

- Python 3.8 or higher
- `pip` for package installation

### **2. Installation**

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### **3. Running the Application**

1.  **Start the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2.  **Open your web browser** and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

### **4. Generating Reports**

1.  **Choose Exam & Report Mode:** Select the exam type (NEET/JEE) and the report type (Unit Test/Chapter-wise).
2.  **Upload Excel Data File:** Upload the Excel file containing the student data.
3.  **Configure Report Settings:** Optionally, specify a test index to generate a report for a specific test.
4.  **Generate Report:** Click the "Generate Report" button to start the process. The generated PDF reports will be saved in the `pdf_results` directory.

---

## 📁 File Structure

```
src/
├── app.py                  # Main Streamlit application
├── data_reader.py          # Reads and parses the input Excel file
├── data_processor.py       # Handles data processing for unit test reports
├── data_processor_2.py     # Handles data processing for chapter-wise reports
├── data_visualizer.py      # Generates all the plots and charts
├── pdf_maker.py            # Creates the PDF reports from HTML templates
├── requirements.txt        # Project dependencies
├── suggestion.txt          # Suggestions for code improvement
├── data/                     # Directory for storing data files
├── pdf_results/            # Output directory for generated PDF reports
├── results/                # Directory for storing generated plots
└── templates/              # HTML and CSS templates for the PDF reports
    ├── assets/
    ├── class/
    ├── student/
    └── subjectWise/
```

---

## 📄 Dependencies

All the required Python libraries are listed in the `requirements.txt` file.

---

## 🖼️ Screenshots

*(You can add screenshots of the application and the generated PDF reports here to give users a visual overview of the tool.)*

---

## 🤝 Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

---

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
