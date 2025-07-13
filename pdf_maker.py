from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from pathlib import Path
import os
import random
from PyPDF2 import PdfMerger

def merge_pdfs_if_exist(new_pdf_path, final_pdf_path):
    """Merge new PDF with existing PDF if it exists"""
    if os.path.exists(final_pdf_path):
        merger = PdfMerger()
        merger.append(final_pdf_path)
        merger.append(new_pdf_path)
        merger.write(final_pdf_path)
        merger.close()
        os.remove(new_pdf_path)
    else:
        os.rename(new_pdf_path, final_pdf_path)

# Setup paths
base_dir = Path(__file__).parent
output_dir = base_dir / "pdf_results"
os.makedirs(output_dir, exist_ok=True)

class BaseReport:
    """Base class for all report types"""
    
    def __init__(self, template_name, processor=None, vis=None):
        self.template_name = template_name
        self.processor = processor
        self.vis = vis
        self._setup_template()
    
    def _setup_template(self):
        """Setup Jinja2 template based on template name"""
        template_folders = {
            'class': base_dir / "templates" / "class",
            'subject': base_dir / "templates" / "subjectWise", 
            'student': base_dir / "templates" / "student"
        }
        
        # Determine template folder
        if 'class' in self.template_name:
            self.template_folder = template_folders['class']
        elif 'subject' in self.template_name:
            self.template_folder = template_folders['subject']
        else:
            self.template_folder = template_folders['student']
        
        # Setup Jinja2 environment
        self.env = Environment(loader=FileSystemLoader(self.template_folder))
        self.template = self.env.get_template(self.template_name)
    
    def _standard_response(self, success, error=None, message=None):
        """Standard response format"""
        return {
            "success": success,
            "error": error,
            "message": message
        }
    
    def create_pdf(self, output_path, context, merge=False):
        """Generate PDF from HTML template"""
        print(f"Generating PDF: {output_path}")
        
        temp_path = base_dir / "temp_output.pdf"
        
        # Render HTML and create PDF
        html_content = self.template.render(context)
        HTML(string=html_content, base_url=str(self.template_folder)).write_pdf(str(temp_path))
        
        # Handle final PDF creation
        final_path = Path(output_path)
        if merge:
            merge_pdfs_if_exist(str(temp_path), str(final_path))
        else:
            if final_path.exists():
                final_path.unlink()
            temp_path.rename(final_path)
        
        print(f"PDF saved: {final_path}")
    
    def get_performance_status(self, accuracy):
        """Get performance status based on accuracy"""
        if accuracy >= 90:
            return "Excellent"
        elif accuracy >= 75:
            return "Strong"
        elif accuracy >= 60:
            return "Good"
        elif accuracy >= 45:
            return "Focus"
        else:
            return "Bad"

class StudentReport(BaseReport):
    """Student report generator"""
    
    def __init__(self, processor=None, vis=None, roll_no=None, test_index=0, 
                 total_questions=None, subjects_total=None, subjects=None):
        super().__init__("student_report.html", processor, vis)
        self.roll_no = roll_no
        self.test_index = test_index
        self.total_questions = total_questions
        self.subjects_total = subjects_total
        self.subjects = subjects
    
    def generate_student_insights(self, student_info, transformed_subjects):
        """Generate insights and performance tags for student"""
        overall_accuracy = student_info['overall_accuracy']
        total_correct = student_info['total_correct']
        total_incorrect = student_info['total_incorrect']
        total_attempted = student_info['total_attempted']
        total_unattempted = student_info['total_unattempted']
        
        # Sort subjects by accuracy
        sorted_subjects = sorted(transformed_subjects, key=lambda x: x['accuracy'], reverse=True)
        best_subject = sorted_subjects[0]
        worst_subject = sorted_subjects[-1]
        
        # Generate general insights
        insights = []
        
        # Overall performance insights
        if overall_accuracy >= 85:
            insights.extend([
                f"Excellent accuracy at {overall_accuracy:.1f}%",
                "You're on track for top ranks – maintain consistency"
            ])
        elif overall_accuracy >= 70:
            insights.extend([
                f"Good accuracy ({overall_accuracy:.1f}%) – minor improvements needed",
                "Focus on speed and accuracy balance"
            ])
        else:
            insights.extend([
                f"Accuracy needs improvement ({overall_accuracy:.1f}%)",
                "Strengthen fundamentals with more practice"
            ])
        
        # Attempt rate insights
        if total_unattempted >= 8:
            insights.append(f"Skipped {total_unattempted} questions – improve time management")
        elif total_unattempted <= 2:
            insights.append("Excellent attempt rate – barely skipped anything")
        
        # Error rate insights
        if total_attempted > 0:
            error_rate = total_incorrect / total_attempted
            if error_rate > 0.3:
                insights.append("High error rate – focus on accuracy")
            elif error_rate < 0.15:
                insights.append("Low error rate – shows strong understanding")
        
        # Subject-wise insights
        for subject in transformed_subjects:
            subj_name = subject['subject']
            accuracy = subject['accuracy']
            
            if accuracy >= 90:
                insights.append(f"{subj_name} is your strongest subject")
            elif accuracy >= 75:
                insights.append(f"{subj_name} shows good performance")
            elif accuracy >= 60:
                insights.append(f"{subj_name} needs minor improvements")
            else:
                insights.append(f"{subj_name} requires focused attention")
        
        # Generate performance tags
        tags = set()
        
        # Accuracy tags
        if overall_accuracy >= 85:
            tags.update(["High Performer", "Consistent", "Rank Potential"])
        elif overall_accuracy >= 70:
            tags.update(["Good Performer", "Improving", "Practice More"])
        else:
            tags.update(["Needs Focus", "Retry Zone", "Build Basics"])
        
        # Attempt rate tags
        if total_unattempted == 0:
            tags.add("Complete Attempt")
        elif total_unattempted >= 8:
            tags.add("Time Management")
        
        # Subject tags
        tags.add(f"{best_subject['subject']} Strong")
        if worst_subject['accuracy'] < 60:
            tags.add(f"{worst_subject['subject']} Weak")
        
        return {
            "insights": insights[:6],  # Limit to 6 insights
            "tags": list(tags)[:8]     # Limit to 8 tags
        }
    
    def create_student_report(self, output_path, context):
        """Create individual student report"""
        print(f"Generating student report for: {self.roll_no}")
        
        # Get student basic data
        student_data = self.processor.get_individual_student_data(
            self.roll_no, test_index=self.test_index
        )
        
        if not student_data["success"]:
            return self._standard_response(
                False, error=student_data.get("error"), 
                message="Failed to fetch student data"
            )
        
        student = student_data["data"][0]
        
        # Student details
        student_details = {
            "roll_no": self.roll_no,
            "name": student["NAME"],
            "mob_no": student["MOB.NO"],
            "father_name": student["FATHER NAME"],
            "total_marks": student["TOTAL MARKS"],
            "obtained_marks": student["OBTAINED MARKS"],
            "percentage": round(student["PERCENTAGE"], 2)
        }
        
        # Get class average
        class_average = self.processor.get_class_average(test_index=self.test_index)
        if not class_average["success"]:
            return self._standard_response(
                False, error=class_average.get("error"), 
                message="Failed to fetch class average"
            )
        
        class_avg_data = class_average["data"][0]
        class_average_details = {
            "class_average_percentage": class_avg_data["average_percentage"],
            "class_average_marks": class_avg_data["class_average"]
        }
        
        # Get subject-wise performance
        student_info = self.processor.get_individual_student_info(
            self.roll_no, test_index=self.test_index,
            total_questions=self.total_questions,
            subjects_total=self.subjects_total
        )
        
        if not student_info["success"]:
            return self._standard_response(
                False, error=student_info.get("error"),
                message="Failed to fetch subject info"
            )
        
        student_full_info = student_info["data"][0]
        subject_data = student_full_info["subject_wise"]
        
        # Transform subject data
        subject_mapping = {
            'PHY': 'Physics',
            'CHEM': 'Chemistry',
            'ZOO': 'Zoology',
            'BOT': 'Botany',
            'MATHS': 'Maths'
        }
        
        transformed_subjects = []
        for short_name, data in subject_data.items():
            transformed_subjects.append({
                'subject': subject_mapping.get(short_name, short_name),
                **data,
                'status': self.get_performance_status(data['accuracy'])
            })
        
        # Generate insights
        insights = self.generate_student_insights(student_full_info, transformed_subjects)
        
        # Generate visualizations
        plots = {
            "student_comparison": self.vis.plot_student_comparison(
                self.roll_no, test_index=self.test_index, subjects=self.subjects
            )["fig"],
            "subject_accuracy_radar": self.vis.get_subject_accuracy_radar(
                self.roll_no, test_index=self.test_index,
                total_questions=self.total_questions,
                subjects_total=self.subjects_total
            )["fig"],
            "gauge_chart": self.vis.plot_gauge_chart(
                self.roll_no, test_index=self.test_index
            )["fig"],
            "accuracy_attempt_matrix": self.vis.plot_accuracy_attempt_matrix(
                self.roll_no, test_index=self.test_index,
                total_questions=self.total_questions,
                subjects_total=self.subjects_total
            )["fig"],
            "combined_accuracy_charts": self.vis.plot_combined_accuracy_charts(
                self.roll_no, test_index=self.test_index,
                total_questions=self.total_questions,
                subjects_total=self.subjects_total
            )["fig"],
            "student_progress_line": self.vis.plot_student_progress_line(
                self.roll_no, test_index=self.test_index
            )["fig"],
        }
        
        # Create final context
        context.update({
            "test_index": self.test_index + 1,
            "student_details": student_details,
            "class_average": class_average_details,
            "subject_wise_data": transformed_subjects,
            "insights": insights,
            "plots": plots,
            "date": datetime.now().strftime("%d-%m-%Y")
        })
        
        # Generate PDF
        self.create_pdf(output_path, context, merge=True)
        
        return self._standard_response(True, message="Student report generated successfully")

class ClassReport(BaseReport):
    """Class report generator"""
    
    def __init__(self, processor=None, vis=None, test_index=0, subjects_total=None):
        super().__init__("class_report.html", processor, vis)
        self.test_index = test_index
        self.subjects_total = subjects_total
    
    def create_class_report(self, output_path, context):
        """Create class performance report"""
        print("Generating class report...")
        
        # Get student summary
        student_summary = self.processor.get_student_summary(
            test_index=self.test_index, subjects_total=self.subjects_total
        )
        
        # Get test statistics
        test_stats = self.processor.get_test_statistics(test_index=self.test_index)
        
        if not student_summary['success'] or not test_stats['success']:
            return self._standard_response(
                False, error="Failed to get class data"
            )
        
        students = student_summary['data']
        if not students:
            return self._standard_response(
                False, error="No student data available"
            )
        
        # Generate class visualizations
        plots = {
            'score_distribution': self.vis.plot_score_distribution_bar_chart(
                test_index=self.test_index, subjects_total=self.subjects_total
            )['fig'],
            'pass_fail_donut': self.vis.plot_pass_fail_donut(
                test_index=self.test_index
            )['fig'],
            'subject_accuracy_radar': self.vis.plot_subject_accuracy_radar(
                test_index=self.test_index, subjects_total=self.subjects_total
            )['fig'],
        }
        
        # Create context
        context.update({
            'test_index': self.test_index + 1,
            'students': sorted(students, key=lambda x: x['obtained_marks'], reverse=True),
            'test_stats': test_stats.get('data', [{}])[0],
            'plots': plots,
            'date': datetime.now().strftime("%d-%m-%Y")
        })
        
        # Generate PDF
        self.create_pdf(output_path, context, merge=True)
        
        return self._standard_response(True, message="Class report generated successfully")

class SubjectWiseReport(BaseReport):
    """Subject-wise report generator"""
    
    def __init__(self, processor=None, vis=None, roll_no=None, test_index=0, 
                 total_subjects_questions=None):
        super().__init__("subject_wise_report.html", processor, vis)
        self.roll_no = roll_no
        self.test_index = test_index
        self.total_subjects_questions = total_subjects_questions
    
    def create_subject_wise_report(self, output_path, context):
        """Create subject-wise performance report"""
        print(f"Generating subject-wise report for: {self.roll_no}")
        
        # Get student details
        student_details = self.processor.get_student_details_info(
            student_roll_no=self.roll_no,
            test_index=self.test_index,
            total_questions=self.total_subjects_questions
        )
        
        if not student_details["success"]:
            return self._standard_response(
                False, error=student_details.get("error"),
                message="Failed to fetch student details"
            )
        
        details_data = student_details["data"]
        student_test_details = details_data.get("student_details", [])
        
        # Add performance status to each test
        for test in student_test_details:
            accuracy = test.get("accuracy", 0)
            test["status"] = self.get_performance_status(accuracy)
        
        # Generate subject-specific plots
        plots = {
            "marks_distribution": self.vis.plot_marks(
                self.roll_no, test_index=self.test_index
            )["fig"],
            "accuracy_distribution": self.vis.plot_accuracy(
                self.roll_no, test_index=self.test_index
            )["fig"],
        }
        
        # Create context
        context.update({
            "test_index": self.test_index + 1,
            "student_details": student_test_details,
            "plots": plots,
            "high_accuracy_chapters": details_data.get("high_accuracy_chapters", 0),
            "focus_required_chapters": details_data.get("focus_required_chapters", 0),
            "date": datetime.now().strftime("%d-%m-%Y")
        })
        
        # Generate PDF
        self.create_pdf(output_path, context)
        
        return self._standard_response(True, message="Subject-wise report generated successfully")