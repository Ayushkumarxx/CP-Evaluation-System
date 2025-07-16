import traceback
import streamlit as st
import pandas as pd
from pathlib import Path
import time
from data_reader import DataReader
from data_processor import DataProcessor
from data_processor_2 import SubjectWiseDataProcessor
from data_visualizer import ClassVisualizer, StudentVisualizer, SubjectVisualizer
from pdf_maker import StudentReport, ClassReport, SubjectWiseReport

# Config
pd.set_option('display.max_columns', None)
st.set_page_config(page_title="Career Point Report Generator", layout="centered")

# Constants
EXAM_CONFIGS = {
    "NEET": {
        "subjects": ["Physics", "Chemistry", "Zoology", "Botany"],
        "subject_codes": ["PHY", "CHEM", "ZOO", "BOT"],
        "subjects_total": {"PHY": 45, "CHEM": 45, "ZOO": 45, "BOT": 45},
        "total_questions_per_subject": 45,
        "total_questions": 180
    },
    "JEE": {
        "subjects": ["Physics", "Chemistry", "Maths"],
        "subject_codes": ["PHY", "CHEM", "MATHS"],
        "subjects_total": {"PHY": 25, "CHEM": 25, "MATHS": 25},
        "total_questions_per_subject": 25,
        "total_questions": 75
    }
}

def get_exam_config(exam_type):
    """Get exam configuration based on type"""
    return EXAM_CONFIGS[exam_type]

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp location"""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / "temp_uploaded_file.xlsx"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return file_path

def get_batch_info(test_df):
    """Extract batch information from test dataframe"""
    if "BATCH" not in test_df.columns:
        st.error("❌ 'BATCH' column not found.")
        st.stop()
    
    batch_values = test_df["BATCH"].dropna().astype(str).unique()
    if len(batch_values) == 1:
        batch_name = batch_values[0]
        st.success(f"✅ Batch Detected: `{batch_name}`")
    elif len(batch_values) > 1:
        batch_name = batch_values[0]
        st.warning(f"⚠️ Multiple batches found. Defaulting to: `{batch_name}`")
    else:
        st.error("❌ No valid batch found in 'BATCH' column.")
        st.stop()
    
    return batch_name

def get_student_data(test_df):
    """Extract student data from test dataframe"""
    if "ROLL.NO" not in test_df.columns:
        st.error("❌ 'ROLL.NO' column missing in the test data.")
        st.stop()
    
    if "MOB.NO" not in test_df.columns:
        st.error("❌ 'MOB.NO' column missing in the test data.")
        st.stop()
    
    students = []
    for _, row in test_df.iterrows():
        if pd.notna(row["ROLL.NO"]) and pd.notna(row["MOB.NO"]):
            students.append({
                "roll_no": str(row["ROLL.NO"]),
                "mob_no": str(row["MOB.NO"])
            })
    
    return students

def create_output_directory(report_type, batch_name, exam_type):
    """Create output directory structured as pdf_results/<exam_type>/<report_type>/<batch_name>"""
    base_result_dir = Path("pdf_results") / exam_type.upper()
    report_folder = "Student Reports" if report_type == "Unit Test Report" else "Chapter-wise Reports"
    result_dir = base_result_dir / report_folder / batch_name
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def generate_unit_test_reports(data_lists, students, batch_name, exam_config, exam_type, all_tests=False ):
    """Generate unit test reports"""
    processor = DataProcessor(data_lists)
    result_dir = create_output_directory("Unit Test Report", batch_name, exam_type)
    
    # Generate class report
    class_vis = ClassVisualizer(processor=processor)
    class_report = ClassReport(
        processor=processor,
        vis=class_vis,
        test_index=len(data_lists) - 1,  # Always use last test for class report
        subjects_total=exam_config["subjects_total"]
    )
    class_report.create_class_report(
        result_dir / "class_report.pdf",
        {"title": "Class Report"}
    )
    
    # Generate student reports
    total_students = len(students)
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()
    
    for i, student in enumerate(students, 1):
        if all_tests:
            # Generate reports for all tests in a single PDF
            for test_idx in range(len(data_lists)):
                vis = StudentVisualizer(processor=processor)
                student_report = StudentReport(
                    processor=processor,
                    vis=vis,
                    roll_no=student["roll_no"],
                    test_index=test_idx,
                    total_questions=exam_config["total_questions"],
                    subjects_total=exam_config["subjects_total"],
                    subjects=exam_config["subject_codes"]
                )
                
                output_path = result_dir / f"{student["roll_no"]}_{student["mob_no"]}_UT.pdf"
                student_report.create_student_report(output_path, {})
        else:
            # Generate report for last test only
            vis = StudentVisualizer(processor=processor)
            student_report = StudentReport(
                processor=processor,
                vis=vis,
                roll_no=student["roll_no"],
                test_index=len(data_lists) - 1,
                total_questions=exam_config["total_questions"],
                subjects_total=exam_config["subjects_total"],
                subjects=exam_config["subject_codes"]
            )
            
            output_path = result_dir / f"{student["roll_no"]}_{student["mob_no"]}_UT.pdf"
            student_report.create_student_report(output_path, {})
        
        # Update progress
        elapsed = time.time() - start_time
        avg = elapsed / i
        remaining = total_students - i
        status_text.text(
            f"📄 Generated {i}/{total_students} Student Reports | ⏳ Remaining: {remaining} | "
            f"ETA: {avg * remaining:.1f} sec"
        )
        progress_bar.progress(i / total_students)

def generate_chapter_wise_reports(data_lists, students, batch_name, exam_config, exam_type, selected_subject):
    """Generate chapter-wise reports"""
    processor = SubjectWiseDataProcessor(data_lists)
    result_dir = create_output_directory("Chapter-wise Report", batch_name, exam_type)
    
    total_students = len(students)
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()
    
    for i, student in enumerate(students, 1):
        vis = SubjectVisualizer(processor=processor)
        subject_report = SubjectWiseReport(
            processor=processor,
            vis=vis,
            roll_no=student["roll_no"],
            test_index=len(data_lists) - 1,  # Always use last test
            total_subjects_questions=exam_config["total_questions_per_subject"]
        )
        
        output_path = result_dir / f"{student["roll_no"]}_{student["mob_no"]}_CT.pdf"
        subject_report.create_subject_wise_report(
            output_path,
            {"title": "Chapter-wise Report", "subject": selected_subject.upper()}
        )
        
        # Update progress
        elapsed = time.time() - start_time
        avg = elapsed / i
        remaining = total_students - i
        status_text.text(
            f"📄 Generated {i}/{total_students} Chapter Reports | ⏳ Remaining: {remaining} | "
            f"ETA: {avg * remaining:.1f} sec"
        )
        progress_bar.progress(i / total_students)

# Main UI
st.title("📊 Career Point Report Generator")
st.markdown("Generate comprehensive student performance reports in a few easy steps.")
st.markdown("---")

# Step 1: Exam and Report Type Selection
st.header("🔍 Step 1: Choose Exam & Report Mode")
st.caption("Please select your exam type and the kind of report you want to generate.")

col1, col2 = st.columns(2)

with col1:
    exam_type = st.selectbox(
        "📘 Select Exam Type", 
        ["NEET", "JEE"], 
        help="Choose NEET for medical or JEE for engineering."
    )

with col2:
    report_type = st.radio(
        "📄 Select Report Type", 
        ["Unit Test Report", "Chapter-wise Report"],
        help="Choose 'Unit Test Report' for individual & class reports. Choose 'Chapter-wise Report' for detailed subject-based reports."
    )

# Get exam configuration
exam_config = get_exam_config(exam_type)

# Conditional inputs based on report type
selected_subject = None
all_tests = False

if report_type == "Chapter-wise Report":
    selected_subject = st.selectbox(
        "🧪 Select Subject", 
        exam_config["subjects"],
        help="Choose the subject for which the report should be generated."
    )
else:
    # Unit Test Report options
    st.subheader("📋 Report Options")
    all_tests = st.checkbox(
        "📚 Generate All Tests Report", 
        value=False,
        help="Check to generate reports for all tests in a single PDF. Unchecked will generate report for last test only."
    )
    
    if not all_tests:
        st.info("📌 Will generate report for the last test only (default)")

st.markdown("---")

# Step 2: File Upload
st.header("📂 Step 2: Upload Excel Data File")
uploaded_file = st.file_uploader(
    "📤 Upload Your Excel File (.xlsx)",
    type=["xlsx"],
    help="The Excel file should contain columns like ROLL.NO, MOB.NO, BATCH, and scores from different tests."
)
st.caption("🔍 Tip: Ensure your file has ROLL.NO, MOB.NO, and BATCH columns.")

st.markdown("---")

# Generate Button
if st.button("🚀 Generate Report", type="primary"):
    if not uploaded_file:
        st.error("❌ Please upload an Excel file before proceeding.")
    else:
        with st.spinner("🔄 Processing your report. Please wait..."):
            try:
                # Save uploaded file
                file_path = save_uploaded_file(uploaded_file)
                
                # Read Excel data
                data_reader = DataReader("temp_uploaded_file.xlsx")
                data_lists = data_reader.df_lists
                
                if not data_lists:
                    st.error("⚠️ No data found in the uploaded file.")
                    st.stop()
                
                # Get last test data for batch and student info
                last_test_df = data_lists[-1]
                batch_name = get_batch_info(last_test_df)
                students = get_student_data(last_test_df)
                
                if not students:
                    st.error("❌ No valid student data found.")
                    st.stop()
                
                st.info(f"📊 Found {len(students)} students in batch: {batch_name}")
                st.info(f"📈 Total tests available: {len(data_lists)}")
                
                # Generate reports based on type
                if report_type == "Unit Test Report":
                    generate_unit_test_reports(data_lists, students, batch_name, exam_config, exam_type, all_tests)
                else:
                    generate_chapter_wise_reports(data_lists, students, batch_name, exam_config, exam_type, selected_subject)
                
                st.success("🎉 All reports generated successfully!")
                
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
                st.code(traceback.format_exc())