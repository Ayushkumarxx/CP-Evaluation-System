import pandas as pd
import traceback
import numpy as np


class BaseDataProcessor:
    """Base class for data processing operations"""
    
    def __init__(self, dfs):
        """Initialize with list of DataFrames"""
        self.dfs = dfs if dfs else []
        self.cleaned_dfs = self.clean_data()

    def clean_data(self):
        """Clean and preprocess each DataFrame"""
        try:
            if not self.dfs:
                return []
                
            cleaned_dfs = []
            for df in self.dfs:
                if df is None or df.empty:
                    continue
                    
                # Replace 'A' with 0 and convert data types
                df = df.replace('A', 0).convert_dtypes()
                
                # Clean percentage column
                if 'PERCENTAGE' in df.columns:
                    df['PERCENTAGE'] = pd.to_numeric(df['PERCENTAGE'], errors='coerce').fillna(0) * 100
                
                # Clean obtained marks column
                if 'OBTAINED MARKS' in df.columns:
                    df['OBTAINED MARKS'] = pd.to_numeric(df['OBTAINED MARKS'], errors='coerce').fillna(0).astype(int)
                    df = df.sort_values('PERCENTAGE', ascending=False).reset_index(drop=True)
                
                cleaned_dfs.append(df)
           
            # Convert string columns to numeric where possible
            for i, df in enumerate(cleaned_dfs):
                cleaned_dfs[i] = df.apply(pd.to_numeric, errors='ignore')
                
            return cleaned_dfs
            
        except Exception as e:
            print(f"Error cleaning data: {e}")
            traceback.print_exc()
            return []

    def _get_standard_response(self, success, data=None, message="", error=None):
        """Standard response structure for all methods"""
        return {
            'success': success,
            'data': data if data is not None else [],
            'message': message,
            'error': str(error) if error else None
        }


class TestStatistics(BaseDataProcessor):
    """Handle test-level statistical operations"""    
    def get_class_average(self, test_index=None): 
        """Get class average for specific test or all tests"""
        try:
            if not self.cleaned_dfs:
                return self._get_standard_response(False, message="No data available")
            
            results = []
            test_indices = [test_index] if test_index is not None else range(len(self.cleaned_dfs))
            
            for i in test_indices:
                if i >= len(self.cleaned_dfs):
                    continue
                    
                df = self.cleaned_dfs[i]
                if df.empty or 'OBTAINED MARKS' not in df.columns:
                    continue
                
                result = {
                    'test_index': i,
                    'class_average': round(float(df['OBTAINED MARKS'].mean()))
                }
                
                # Check if 'PERCENTAGE' column exists and add its average
                if 'PERCENTAGE' in df.columns:
                    result['average_percentage'] = round(float(df['PERCENTAGE'].mean()), 2)
                
                results.append(result)
            
            return self._get_standard_response(True, results, "Class averages calculated successfully")
            
        except Exception as e:
            return self._get_standard_response(False, error=e, message="Error calculating class average")

    def get_test_statistics(self, test_index=None):
        """Get statistical summary for specific test or all tests"""
        try:
            if not self.cleaned_dfs:
                return self._get_standard_response(False, message="No data available")
                
            results = []
            test_indices = [test_index] if test_index is not None else range(len(self.cleaned_dfs))
            
            for i in test_indices:
                if i >= len(self.cleaned_dfs):
                    continue
                    
                df = self.cleaned_dfs[i]
                if 'OBTAINED MARKS' not in df.columns or df.empty:
                    continue
                    
                marks = df['OBTAINED MARKS']
                stats = {
                    'test_index': i,
                    'mean': float(marks.mean()),
                    'median': float(marks.median()),
                    'std_dev': float(marks.std()),
                    'min': int(marks.min()),
                    'max': int(marks.max()),
                    'total_students': len(df)
                }
                results.append(stats)
            
            return self._get_standard_response(True, results, "Test statistics calculated successfully")
            
        except Exception as e:
            return self._get_standard_response(False, error=e, message="Error calculating test statistics")
    
    def get_marks_distribution_and_subject_stats(self, test_index=None, subjects_total=None):
        """
        Generate marks distribution and subject-wise stats (accuracy, attempt rate, unattempt rate).
        Distribution is evenly split into 4–6 bins from 0 to total marks, including total marks.
        """
        try:
            if subjects_total is None:
                subjects_total = {"PHY": 45, "CHEM": 45, "ZOO": 45, "BOT": 45}

            total_questions = sum(subjects_total.values())
            results = []

            if not self.cleaned_dfs:
                return self._get_standard_response(False, message="No data available")

            test_indices = [test_index] if test_index is not None else range(len(self.cleaned_dfs))

            for i in test_indices:
                if i >= len(self.cleaned_dfs):
                    continue

                df = self.cleaned_dfs[i]
                if df.empty or 'OBTAINED MARKS' not in df.columns:
                    continue

                # Ensure OBTAINED MARKS are numeric
                df['OBTAINED MARKS'] = pd.to_numeric(df['OBTAINED MARKS'], errors='coerce').fillna(0)
                obtained_marks = df['OBTAINED MARKS']

                # Determine total marks (fallback to total questions)
                if 'TOTAL MARKS' in df.columns:
                    total_marks = pd.to_numeric(df['TOTAL MARKS'], errors='coerce').dropna().iloc[0]
                else:
                    total_marks = total_questions

                # Generate 4–6 bins based on total marks
                bin_count = min(6, max(4, total_marks // 100))
                bin_edges = np.linspace(0, total_marks, bin_count + 1).astype(int)

                # Labels like "0-119", ..., "600-720"
                bin_labels = [
                    f"{bin_edges[j]}-{bin_edges[j+1] - 1 if j < len(bin_edges)-2 else bin_edges[j+1]}"
                    for j in range(len(bin_edges) - 1)
                ]

                # Assign students to bins
                df['marks_bin'] = pd.cut(
                    obtained_marks,
                    bins=bin_edges,
                    labels=bin_labels,
                    include_lowest=True,
                    right=True  # include right edge, so 720 is covered
                )

                # Create distribution DataFrame
                marks_distribution_df = df['marks_bin'].value_counts().sort_index().reset_index()
                marks_distribution_df.columns = ['marks_range', 'student_count']

                # Subject-wise stats
                subject_stats = {}
                for subj, total_subj_q in subjects_total.items():
                    correct_col = f'{subj} CORRECT'
                    incorrect_col = f'{subj} INCORRECT'

                    if correct_col not in df.columns or incorrect_col not in df.columns:
                        continue

                    correct = pd.to_numeric(df[correct_col], errors='coerce').fillna(0)
                    incorrect = pd.to_numeric(df[incorrect_col], errors='coerce').fillna(0)
                    attempted = correct + incorrect
                    unattempted = total_subj_q - attempted

                    accuracy = np.where(attempted > 0, (correct / attempted) * 100, 0)
                    attempt_rate = (attempted / total_subj_q) * 100
                    unattempt_rate = (unattempted / total_subj_q) * 100

                    subject_stats[subj] = {
                        'average_accuracy': round(accuracy.mean(), 2),
                        'average_attempt_rate': round(attempt_rate.mean(), 2),
                        'average_unattempt_rate': round(unattempt_rate.mean(), 2),
                    }

                # Combined stats
                correct_cols = [col for col in df.columns if 'CORRECT' in col]
                incorrect_cols = [col for col in df.columns if 'INCORRECT' in col]
                total_correct = df[correct_cols].sum().sum()
                total_incorrect = df[incorrect_cols].sum().sum()
                total_attempted = total_correct + total_incorrect
                total_possible = total_questions * len(df)

                combined_stats = {
                    'overall_accuracy': round((total_correct / total_attempted) * 100, 2) if total_attempted > 0 else 0,
                    'overall_attempt_rate': round((total_attempted / total_possible) * 100, 2),
                    'overall_unattempt_rate': round(((total_possible - total_attempted) / total_possible) * 100, 2),
                    'min_percentage': round((obtained_marks.min() / total_marks) * 100, 2),
                    'max_percentage': round((obtained_marks.max() / total_marks) * 100, 2)
                }

                results.append({
                    'test_index': i,
                    'marks_distribution': marks_distribution_df.to_dict(orient='records'),
                    'subject_wise_stats': subject_stats,
                    'combined_stats': combined_stats
                })

            return self._get_standard_response(True, results, "Marks distribution and subject-wise stats calculated successfully")

        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._get_standard_response(False, error=e, message="Error generating statistics")




class SubjectAnalysis(BaseDataProcessor):
    """Handle subject-wise analysis operations"""
    
    def get_subject_wise_average(self, subjects, test_index=None):
        """Get average marks per subject across specific test or all tests, in simple dict format"""
        try:
            if not self.cleaned_dfs or not subjects:
                return self._get_standard_response(False, message="No data or subjects available")

            subject_totals = {}
            dfs = [self.cleaned_dfs[test_index]] if test_index is not None else self.cleaned_dfs

            for df in dfs:
                if df.empty:
                    continue

                for subject in subjects:
                    if subject in df.columns:
                        df[subject] = pd.to_numeric(df[subject], errors='coerce').fillna(0)
                        if subject not in subject_totals:
                            subject_totals[subject] = []
                        subject_totals[subject].extend(df[subject].tolist())

            avg_dict = {}
            for subject, marks in subject_totals.items():
                if marks:
                    avg_dict[subject] = round(float(pd.Series(marks).mean()))

            avg_dict["test_scope"] = "specific" if test_index is not None else "all"

            return self._get_standard_response(True, [avg_dict], "Subject-wise averages calculated successfully")

        except Exception as e:
            return self._get_standard_response(False, error=str(e), message="Error in subject-wise analysis")


class StudentAnalysis(BaseDataProcessor):
    """Handle student-level analysis operations""" 
    def get_student_summary(self, test_index=None, subjects_total=None):
        """
        Get a summary for each student for a specific test or across all tests.
        """
        try:
            if not self.cleaned_dfs:
                return self._get_standard_response(False, message="No data available")

            if subjects_total is None:
                subjects_total = {"PHY": 45, "CHEM": 45, "ZOO": 45, "BOT": 45}
            
            total_questions = sum(subjects_total.values())

            results = []
            test_indices = [test_index] if test_index is not None else range(len(self.cleaned_dfs))

            for i in test_indices:
                if i >= len(self.cleaned_dfs):
                    continue
                
                df = self.cleaned_dfs[i]
                if df.empty or 'ROLL.NO' not in df.columns or 'NAME' not in df.columns:
                    continue

                for _, row in df.iterrows():
                    student_roll_no = row['ROLL.NO']
                    student_name = row['NAME']
                    
                    subject_wise = {}
                    total_correct = 0
                    total_incorrect = 0

                    # Calculate subject-wise data
                    for subject in subjects_total.keys():
                        correct_key = f"{subject} CORRECT"
                        incorrect_key = f"{subject} INCORRECT"

                        correct = row.get(correct_key, 0)
                        incorrect = row.get(incorrect_key, 0)
                        attempted = correct + incorrect
                        unattempted = subjects_total[subject] - attempted
                        
                        total_correct += correct
                        total_incorrect += incorrect

                        subject_wise[subject] = {
                            "correct": correct,
                            "incorrect": incorrect,
                            "attempted": attempted,
                            "unattempted": unattempted,
                            "total_questions": subjects_total[subject],
                            "accuracy": round((correct / attempted) * 100, 2) if attempted > 0 else 0
                        }
                    
                    total_attempted = total_correct + total_incorrect
                    total_unattempted = total_questions - total_attempted
                    
                    student_summary = {
                        'test_index': i,
                        'roll_no': student_roll_no,
                        'name': student_name,
                        'obtained_marks': row.get('OBTAINED MARKS', 0),
                        'total_marks': row.get('TOTAL MARKS', total_questions * 4),
                        'total_correct': total_correct,
                        'total_incorrect': total_incorrect,
                        'total_attempted': total_attempted,
                        'total_unattempted': total_unattempted,
                        'overall_accuracy': round((total_correct / total_attempted) * 100, 2) if total_attempted > 0 else 0,
                        'subjects': subject_wise
                    }
                    results.append(student_summary)

            return self._get_standard_response(True, results, "Student summary generated successfully")

        except Exception as e:
            traceback.print_exc()
            return self._get_standard_response(False, error=e, message="Error generating student summary")

    def get_individual_student_data(self, student_roll_no, test_index=None):
        """Get individual student data for specific test or all tests"""
        try:
            if not self.cleaned_dfs or not student_roll_no:
                return self._get_standard_response(False, message="No data or student roll number provided")
                
            results = []
            test_indices = [test_index] if test_index is not None else range(len(self.cleaned_dfs))
            
            for i in test_indices:
                if i >= len(self.cleaned_dfs):
                    continue
                
                df = self.cleaned_dfs[i]
                if df.empty or 'ROLL.NO' not in df.columns:
                    continue
                    
                match = df[df['ROLL.NO'].str.strip().str.lower() == student_roll_no.strip().lower()]
                if not match.empty:
                    row = match.iloc[0]
                    student_data = row.to_dict()
                    student_data['test_index'] = test_index if test_index is not None else i
                    results.append(student_data)
            
            return self._get_standard_response(True, results, "Student data retrieved successfully")
            
        except Exception as e:
            return self._get_standard_response(False, error=e, message="Error fetching student data")

    def get_individual_student_info(self, student_roll_no, total_questions=180, subjects_total=None, test_index=None):
        """Get detailed student information with subject-wise breakdown"""
        try:
            if subjects_total is None:
                subjects_total = {"PHY": 45, "CHEM": 45, "ZOO": 45, "BOT": 45}
                
            student_data_response = self.get_individual_student_data(student_roll_no, test_index)
            if not student_data_response['success']:
                return student_data_response
                
            results = []
            for data in student_data_response['data']:
                subject_wise = {}
                total_correct = 0
                total_incorrect = 0
                
                # Calculate totals
                for key, value in data.items():
                    if "INCORRECT" in key and isinstance(value, (int, float)):
                        total_incorrect += value
                    elif "CORRECT" in key and isinstance(value, (int, float)):
                        total_correct += value
                        
                # Calculate subject-wise data
                for subject in subjects_total.keys():
                    correct_key = f"{subject} CORRECT"
                    incorrect_key = f"{subject} INCORRECT"

                    correct = data.get(correct_key, 0)
                    incorrect = data.get(incorrect_key, 0)
                    attempted = correct + incorrect
                    unattempted = subjects_total[subject] - attempted

                    subject_wise[subject] = {
                        "correct": correct,
                        "incorrect": incorrect,
                        "attempted": attempted,
                        "unattempted": unattempted,
                        "total": subjects_total[subject],
                        "accuracy": round((correct / attempted) * 100, 2) if attempted > 0 else 0
                    }
                    
                total_attempted = total_correct + total_incorrect
                total_unattempted = total_questions - total_attempted
                
                results.append({
                    'test_index': data['test_index'],
                    'student_name': data['NAME'],
                    'student_roll_no': student_roll_no,
                    'total_correct': total_correct,
                    'total_incorrect': total_incorrect,
                    'total_marks': data['OBTAINED MARKS'],
                    'total_attempted': total_attempted,
                    'total_unattempted': total_unattempted,
                    'overall_accuracy': round((total_correct / total_attempted) * 100, 2) if total_attempted > 0 else 0,
                    'subject_wise': subject_wise
                })
            
            return self._get_standard_response(True, results, "Student information retrieved successfully")
            
        except Exception as e:
            return self._get_standard_response(False, error=e, message="Error fetching student information")
            
    def get_student_progress(self, student_roll_no, test_index=None):
        """Get student's marks across all tests for progress tracking"""
        try:
            if not self.cleaned_dfs or not student_roll_no :
                return self._get_standard_response(False, message="No data or student roll number provided")
                
            progress = []
            for i, df in enumerate(self.cleaned_dfs[:test_index + 1]):
                if df.empty or 'ROLL.NO' not in df.columns or 'OBTAINED MARKS' not in df.columns:
                    progress.append({'test_index': i, 'marks': None, 'total_marks': None,  'status': 'no_data'})
                    continue
                    
                row = df[df['ROLL.NO'].str.strip().str.lower() == student_roll_no.strip().lower()]
                if not row.empty:
                    marks = int(row['OBTAINED MARKS'].iloc[0])
                    total_marks = int(row['TOTAL MARKS'].iloc[0])
                    progress.append({'test_index': i, 'marks': marks, 'total_marks': total_marks,  'status': 'present'})
                else:
                    progress.append({'test_index': i, 'marks': None, 'total_marks': None, 'status': 'absent'})
            
            return self._get_standard_response(True, progress, "Student progress retrieved successfully")
            
        except Exception as e:
            return self._get_standard_response(False, error=e, message="Error getting student progress")

    
class RankingAnalysis(BaseDataProcessor):
    """Handle ranking and performance comparison operations"""
    def get_student_rank(self, student_roll_no, test_index=None):
        """Get student's rank, percentage, and grade for a specific test or all tests"""
        try:
            if not self.cleaned_dfs:
                return self._get_standard_response(False, message="No data available")

            results = []
            test_indices = [test_index] if test_index is not None else range(len(self.cleaned_dfs))

            for i in test_indices:
                if i >= len(self.cleaned_dfs):
                    continue

                df = self.cleaned_dfs[i]
                if df.empty or 'ROLL.NO' not in df.columns or 'OBTAINED MARKS' not in df.columns:
                    continue

                df = df.reset_index(drop=True)  # Ensure correct indexing
                student_row = df[df['ROLL.NO'].str.strip().str.lower() == student_roll_no.strip().lower()]
                if student_row.empty:
                    continue

                rank = student_row.index[0] + 1
                percentage = student_row.iloc[0].get("PERCENTAGE", 0)

                # Grade logic
                if percentage >= 90:
                    grade = "A+"
                elif percentage >= 80:
                    grade = "A"
                elif percentage >= 70:
                    grade = "B+"
                elif percentage >= 60:
                    grade = "B"
                elif percentage >= 50:
                    grade = "C"
                elif percentage >= 40:
                    grade = "D"
                else:
                    grade = "F"

                results.append({
                    'test_index': i,
                    'rank': rank,
                    'total_students': len(df),
                    'percentage': round(percentage, 1),
                    'grade': grade
                })

            return self._get_standard_response(True, results, "Student ranks retrieved successfully")

        except Exception as e:
            return self._get_standard_response(False, error=e, message="Error getting student ranks")

    def get_pass_fail_summary(self, test_index=None):
        """
        Efficiently returns summary of passed and failed students:
        - Number of students passed and failed
        - List of passed and failed students
        - Pass and fail percentages
        """

        try:
            if not self.cleaned_dfs:
                return self._get_standard_response(False, message="No data available")

            passed, failed = [], []
            dfs = [self.cleaned_dfs[test_index]] if test_index is not None else self.cleaned_dfs

            for idx, df in enumerate(dfs):
                if df.empty or not {'NAME', 'OBTAINED MARKS', 'TOTAL MARKS'}.issubset(df.columns):
                    continue

                test_no = test_index if test_index is not None else idx

                # Calculate pass mark once per test
                total_marks = df['TOTAL MARKS'].iloc[0]
                pass_mark = int(0.4 * total_marks)

                for _, row in df.iterrows():
                    marks = int(row['OBTAINED MARKS'])
                    student_data = {
                        'test_index': test_no,
                        'name': row['NAME'],
                        'roll_no': row.get('ROLL.NO', 'N/A'),
                        'marks': marks
                    }

                    if marks >= pass_mark:
                        student_data['surplus'] = marks - pass_mark
                        passed.append(student_data)
                    else:
                        student_data['deficit'] = pass_mark - marks
                        failed.append(student_data)

            total = len(passed) + len(failed)
            pass_percent = round((len(passed) / total) * 100, 2) if total else 0.0
            fail_percent = round((len(failed) / total) * 100, 2) if total else 0.0

            return self._get_standard_response(
                True,
                {
                    'total_students': total,
                    'passed_count': len(passed),
                    'failed_count': len(failed),
                    'passed_students': passed,
                    'failed_students': failed,
                    'pass_percentage': pass_percent,
                    'fail_percentage': fail_percent
                },
                "Pass/Fail summary generated successfully"
            )

        except Exception as e:
            return self._get_standard_response(False, error=e, message="Error generating pass/fail summary")

class DataProcessor(TestStatistics, SubjectAnalysis, StudentAnalysis, RankingAnalysis):
    """Main data processor class that inherits from all specialized classes"""
    
    def __init__(self, dfs):
        """Initialize the main data processor"""
        super().__init__(dfs)
        
   