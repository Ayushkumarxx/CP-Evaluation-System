import pandas as pd
import traceback
import numpy as np


class BaseSubjectProcessor:
    """Base class for subject-wise data processing operations"""
    
    def __init__(self, dfs):
        """Initialize with list of DataFrames"""
        self.dfs = dfs if dfs else []
        self.cleaned_dfs = self.clean_data()

    def clean_data(self):
        """Clean and preprocess each DataFrame for subject-wise analysis"""
        try:
            if not self.dfs:
                return []
                
            cleaned_dfs = []
            for df in self.dfs:
                if df is None or df.empty:
                    continue
                    
                # Create a copy to avoid modifying original
                df = df.copy()
                
                # Replace 'A' with 0 and convert data types in one pass
                df = df.replace('A', 0)
                
                # Vectorized operations for numeric columns
                numeric_cols = ['PERCENTAGE', 'TOTAL MARKS']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                        if col == 'PERCENTAGE' and df[col].max() <= 1:
                            df[col] *= 100
                        elif col == 'TOTAL MARKS':
                            df[col] = df[col].astype(int)
                
                # Vectorized operations for subject-specific columns
                subject_keywords = ['CORRECT', 'INCORRECT', 'TOTAL']
                subject_cols = [col for col in df.columns 
                               if any(keyword in col.upper() for keyword in subject_keywords)]
                
                if subject_cols:
                    for col in subject_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                
                # Sort by total marks or percentage in descending order
                if 'TOTAL MARKS' in df.columns:
                    df = df.sort_values('TOTAL MARKS', ascending=False).reset_index(drop=True)
                elif 'PERCENTAGE' in df.columns:
                    df = df.sort_values('PERCENTAGE', ascending=False).reset_index(drop=True)
                
                cleaned_dfs.append(df)
           
            # Convert string columns to numeric where possible - batch operation
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


class SubjectWiseStudentAnalysis(BaseSubjectProcessor):
    """Handle subject-wise student analysis operations"""
    
    def get_student_details_info(self, student_roll_no=None, test_index=None, total_questions=45):
        """
        Get detailed student information including personal details,
        obtained marks, accuracy for a single subject, and cumulative accuracy per test.
        """
        try:
            if not self.cleaned_dfs:
                return self._get_standard_response(False, message="No data available")

            student_details = []
            total_correct_so_far = 0
            total_incorrect_so_far = 0
            high_accuracy_chapters = 0
            focus_required_chapters = 0

            end_index = test_index + 1 if test_index is not None else len(self.cleaned_dfs)

            # Pre-normalize student roll number once
            normalized_roll_no = student_roll_no.strip().upper() if student_roll_no else None

            for i in range(end_index):
                df = self.cleaned_dfs[i]
                if df.empty:
                    continue

                # Filter student rows efficiently
                if normalized_roll_no:
                    # Vectorized string operations
                    mask = df['ROLL.NO'].astype(str).str.strip().str.upper() == normalized_roll_no
                    student_rows = df[mask]
                else:
                    student_rows = df

                if student_rows.empty:
                    continue

                # Pre-find correct column once per dataframe
                correct_col = None
                for col in df.columns:
                    if 'CORRECT' in col.upper() and not col.upper().startswith('TOTAL'):
                        correct_col = col
                        break

                # Vectorized operations on the filtered dataframe
                if correct_col:
                    incorrect_col = correct_col.replace('CORRECT', 'INCORRECT')
                    total_col = correct_col.replace('CORRECT', 'TOTAL')
                    
                    # Get all values at once using vectorized operations
                    correct_values = student_rows[correct_col].values
                    incorrect_values = student_rows[incorrect_col].values
                    total_values = student_rows[total_col].values
                    
                    # Calculate attempted and accuracy vectorized
                    attempted_values = correct_values + incorrect_values
                    unattempted_values = total_questions - attempted_values
                    accuracy_values = np.where(attempted_values > 0, 
                                             np.round((correct_values / attempted_values) * 100, 2), 
                                             0)

                # Process each row
                for idx, (_, row) in enumerate(student_rows.iterrows()):
                    details = {
                        'test_index': test_index if test_index is not None else i,
                        'name': row.get('NAME', 'Unknown'),
                        'father_name': row.get('FATHER NAME', 'Unknown'),
                        'mobile_no': row.get('MOB.NO', 'Unknown'),
                        'roll_no': row.get('ROLL.NO', 'Unknown'),
                        'batch': row.get('BATCH', 'Unknown'),
                        'total_marks': row.get('TOTAL MARKS', 0),
                        'percentage': round(row.get('PERCENTAGE', 0), 2),
                        'test_name': row.get('TEST NAME', f'Test {i+1}'),
                    }

                    if correct_col:
                        correct = correct_values[idx]
                        incorrect = incorrect_values[idx]
                        total = total_values[idx]
                        attempted = attempted_values[idx]
                        unattempted = unattempted_values[idx]
                        accuracy = accuracy_values[idx]

                        # Update cumulative totals
                        total_correct_so_far += correct
                        total_incorrect_so_far += incorrect
                        attempted_so_far = total_correct_so_far + total_incorrect_so_far
                        cumulative_accuracy = round((total_correct_so_far / attempted_so_far) * 100, 2) if attempted_so_far > 0 else 0

                        # Update counts
                        if accuracy >= 70:
                            high_accuracy_chapters += 1
                        if accuracy < 50:
                            focus_required_chapters += 1

                        details.update({
                            'correct': int(correct),
                            'incorrect': int(incorrect),
                            'attempted': int(attempted),
                            'unattempted': int(unattempted),
                            'accuracy': float(accuracy),
                            'total_subject_marks': int(total),
                            'cumulative_accuracy': cumulative_accuracy
                        })

                    student_details.append(details)

            result = {
                'student_details': student_details,
                'high_accuracy_chapters': high_accuracy_chapters,
                'focus_required_chapters': focus_required_chapters
            }

            return self._get_standard_response(True, result, "Student details retrieved successfully")

        except Exception as e:
            return self._get_standard_response(False, error=e, message="Error retrieving student details")

    def get_student_progress_tracking(self, student_roll_no, test_index=None):
        try:
            if not self.cleaned_dfs or not student_roll_no:
                return self._get_standard_response(False, message="No data or student roll number provided")

            accuracy_list = []
            subject_total_marks_list = []
            
            end_index = test_index + 1 if test_index is not None else len(self.cleaned_dfs)
            
            # Pre-normalize student roll number once
            normalized_roll_no = student_roll_no.strip().upper()

            for i in range(end_index):
                df = self.cleaned_dfs[i]

                if df.empty or 'ROLL.NO' not in df.columns:
                    continue

                # Vectorized student lookup
                mask = df['ROLL.NO'].astype(str).str.strip().str.upper() == normalized_roll_no
                student_rows = df[mask]
                
                if student_rows.empty:
                    continue

                row = student_rows.iloc[0]

                # Find correct column once per dataframe
                correct_col = None
                for col in df.columns:
                    col_upper = col.upper()
                    if 'CORRECT' in col_upper and not col_upper.startswith('TOTAL'):
                        correct_col = col
                        break

                if correct_col:
                    incorrect_col = correct_col.replace('CORRECT', 'INCORRECT')
                    total_col = correct_col.replace('CORRECT', 'TOTAL')

                    correct = row.get(correct_col, 0)
                    incorrect = row.get(incorrect_col, 0)
                    total = row.get(total_col, 0)

                    attempted = correct + incorrect
                    accuracy = round((correct / attempted) * 100, 2) if attempted > 0 else 0

                    accuracy_list.append(accuracy)
                    subject_total_marks_list.append(total)

            return self._get_standard_response(True, {
                'accuracy_list': accuracy_list,
                'subject_total_marks_list': subject_total_marks_list
            }, "Progress retrieved")

        except Exception as e:
            return self._get_standard_response(False, error=e, message="Something went wrong")


class SubjectWiseDataProcessor(SubjectWiseStudentAnalysis):
    """Main subject-wise data processor class that inherits from all specialized classes"""
    
    def __init__(self, dfs):
        """Initialize the main subject-wise data processor"""
        super().__init__(dfs)