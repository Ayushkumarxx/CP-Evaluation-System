import base64
from io import BytesIO
from math import pi
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import plotly.graph_objects as go
from functools import lru_cache
import gc

warnings.filterwarnings('ignore')
# Set matplotlib to use a faster backend
matplotlib.use('Agg')  # Non-interactive backend

# Matplotlib settings - Pre-configured for better performance
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial, sans-serif',
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#222222',
    'grid.color': '#dddddd',
    'grid.linestyle': '--',
    'lines.linewidth': 2.2,
    'lines.markersize': 6,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 100,
})


class DataVisualizerHelper:
    """
    Base helper class for data visualization with optimized performance.
    
    Performance Optimizations:
    - Uses __slots__ for memory efficiency
    - Caches color dictionary for faster lookups
    - Optimized figure saving with memory cleanup
    - Pre-computed result directory path
    
    """
    
    __slots__ = ('processor', 'result_dir', '_colors_cache')
    
    def __init__(self, processor):
        self.processor = processor
       

        # Pre-cache colors for faster access
        self._colors_cache = {
            'blue': '#3B82F6', 'orange': '#F97316', 'green': '#10B981',
            'red': '#EF4444', 'purple': '#8B5CF6', 'brown': '#A97155',
            'pink': '#EC4899', 'gray': '#6B7280', 'olive': '#A3B763',
            'cyan': '#22D3EE', 'black': '#111827', 'navy': '#1E3A8A',
            'mustard': '#EAB308', 'teal': '#14B8A6', 'maroon': '#881337',
            'indigo': '#6366F1', 'beige': '#F5F5DC', 'lime': '#84CC16',
            'slate': '#94A3B8', 'rose': '#F43F5E', 'gold': '#FACC15',
            'steel': '#64748B'
        }

    @lru_cache(maxsize=32)
    def get_color(self, color_name):
        """Cached color lookup for better performance."""
        return self._colors_cache.get(color_name, '#333333')
    
    def fig_to_base64(self, fig, backend='matplotlib'):
        """Convert a Matplotlib or Plotly figure to base64 string."""
        buf = BytesIO()
        
        if backend == 'matplotlib':
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
        elif backend == 'plotly':
            fig.write_image(buf, format='png', width=600, height=600, scale=2)
            del fig

        gc.collect()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')



    def _standard_response(self, success, message="", error=None, fig=None):
        """Optimized response structure - removed data field for performance."""
        return {
            'success': success,
            'message': message,
            'error': str(error) if error else None,
            'fig': fig
        }


class StudentVisualizer(DataVisualizerHelper):
    """
    Student-specific visualization class with performance optimizations.

    """

    __slots__ = ('_data_cache',)
    
    def __init__(self, processor):
        super().__init__(processor)
        self._data_cache = {}  # Cache for frequently accessed data

    def _get_student_data(self, roll_no, test_index=0, subjects_total=None, total_questions=None):
        """Optimized data fetching with caching."""
        cache_key = f"{roll_no}_{test_index}"
        
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        student_data = self.processor.get_individual_student_info(
            student_roll_no=roll_no, test_index=test_index, subjects_total=subjects_total,
            total_questions=total_questions
        )
        
        if not student_data["success"]:
            return None, self._standard_response(
                False, error=student_data["error"], message=student_data["message"]
            )
        
        result = student_data["data"][0], student_data
        self._data_cache[cache_key] = result
        return result

    def plot_combined_accuracy_charts(self, roll_no, test_index=0, subjects_total=None, total_questions=None):
        """Optimized donut chart generation."""
        data, student_data = self._get_student_data(roll_no, test_index, subjects_total, total_questions)
        if not data:
            return student_data

        # Pre-extract values for better performance
        correct = data.get("total_correct", 0)
        incorrect = data.get("total_incorrect", 0)
        unattempted = data.get("total_unattempted", 0)
        accuracy = data.get("overall_accuracy", 0)

        values = [correct, incorrect, unattempted]
        labels = ['Correct', 'Incorrect', 'Unattempted']
        colors = [self.get_color('green'), self.get_color('red'), self.get_color('slate')]
        total = sum(values)

        # Optimized figure creation
        fig, ax = plt.subplots(figsize=(6, 6))

        # Inline function for better performance
        def make_label(pct, count):
            return f"({count})"

        wedges, texts, autotexts = ax.pie(
            values, labels=labels,
            autopct=lambda pct: make_label(pct, int(round(pct * total / 100))),
            startangle=90, colors=colors,
            wedgeprops=dict(width=0.3, edgecolor='white'),
            textprops=dict(color="black", fontsize=12, fontweight='bold')
        )

        ax.text(0, 0, f"Accuracy: {accuracy:.1f}%", ha='center', va='center',
                fontsize=12, fontweight='bold', color='black')
        ax.set_title("Accuracy Donut Chart", fontsize=16, fontweight='bold')

        image_base64 = self.fig_to_base64(fig, backend='matplotlib')  
        return self._standard_response(True, message="Donut chart created", fig=image_base64)

    def plot_accuracy_attempt_matrix(self, roll_no, test_index=0, subjects_total=None, total_questions=None):
        """Optimized accuracy vs attempt matrix visualization."""
        data, student_data = self._get_student_data(roll_no, test_index, subjects_total, total_questions)
        if not data:
            return student_data

        subject_data = data.get("subject_wise", {})
        if not subject_data:
            return self._standard_response(False, message="No subject-wise data found.")

        # Pre-compute all values for better performance
        subjects = list(subject_data.keys())
        attempts = [subject_data[sub]["attempted"] for sub in subjects]
        accuracies = [subject_data[sub]["accuracy"] for sub in subjects]
        
        total_questions = [
            subject_data[sub].get("total_questions", 
                               subject_data[sub]["attempted"] + subject_data[sub]["unattempted"])
            for sub in subjects
        ]
        attempt_rates = [(att / total) * 100 if total > 0 else 0 
                        for att, total in zip(attempts, total_questions)]

        # Optimized threshold computation
        attempt_threshold, accuracy_threshold = 55, 65
        
        # Vectorized color and label computation
        colors, labels = [], []
        for att_rate, acc in zip(attempt_rates, accuracies):
            if acc >= accuracy_threshold and att_rate >= attempt_threshold:
                colors.append('#2E8B57')
                labels.append('Optimal')
            elif acc >= accuracy_threshold:
                colors.append('#4169E1')
                labels.append('Potential')
            elif att_rate >= attempt_threshold:
                colors.append('#FF8C00')
                labels.append('Struggler')
            else:
                colors.append('#DC143C')
                labels.append('Neglected')

        # Optimized plotting
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)

        # Batch scatter plotting
        for i, (x, y, color, subject) in enumerate(zip(attempt_rates, accuracies, colors, subjects)):
            ax.scatter(x, y, color=color, s=200, edgecolors='black', 
                      linewidth=2, alpha=0.9, zorder=4)
            ax.annotate(subject, (x, y), xytext=(6, 6), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       zorder=6)

        # Threshold lines and grid
        ax.axhline(accuracy_threshold, linestyle='--', color='gray', linewidth=2, alpha=0.7, zorder=2)
        ax.axvline(attempt_threshold, linestyle='--', color='gray', linewidth=2, alpha=0.7, zorder=2)
        ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.7, zorder=1)

        # Optimized quadrant labels
        quadrant_specs = [
            (75, 75, "Optimal\nHigh Accuracy & High Attempts", '#2E8B57', 'lightgreen'),
            (25, 75, "Potential\nHigh Accuracy & Low Attempts", '#4169E1', 'lightblue'),
            (75, 25, "Struggler\nLow Accuracy & High Attempts", '#FF8C00', 'moccasin'),
            (25, 25, "Neglected\nLow Accuracy & Low Attempts", '#DC143C', 'mistyrose'),
        ]
        
        for x, y, text, color, facecolor in quadrant_specs:
            ax.text(x, y, text, fontsize=6, fontweight='bold', color=color,
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=facecolor, alpha=0.8),
                   zorder=0)

        # Styling
        ax.set_xlabel("Attempt Rate (%)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
        ax.set_title("Accuracy vs Attempt", fontsize=16, fontweight='bold', pad=20)
        ax.tick_params(axis='both', labelsize=9)
        
        # Batch tick styling
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            
        ax.set_facecolor('#f8f9fa')
        plt.tight_layout()

 
        image_base64 = self.fig_to_base64(fig, backend='matplotlib')  
        return self._standard_response(True, message="Accuracy matrix plotted", fig=image_base64)

    def get_subject_accuracy_radar(self, roll_no, test_index=0, subjects_total=None, total_questions=None):
        """Optimized radar chart generation."""
        data, student_data = self._get_student_data(roll_no, test_index, subjects_total, total_questions)
        if not data:
            return student_data

        subject_data = data["subject_wise"]
        subjects = list(subject_data.keys())

        # Vectorized computation
        accuracies = []
        attempts = []

        for subj in subjects:
            subj_info = subject_data[subj]
            acc = subj_info.get("accuracy", 0)
            attempted = subj_info.get("attempted", 0)
            unattempted = subj_info.get("unattempted", 0)
            
            total = attempted + unattempted
            attempt_percent = (attempted / total * 100) if total > 0 else 0
            
            accuracies.append(acc)
            attempts.append(attempt_percent)

        # Pre-compute angles for better performance
        n_subjects = len(subjects)
        angles = [n / float(n_subjects) * 2 * pi for n in range(n_subjects)]
        
        # Close the radar chart
        accuracies += accuracies[:1]
        attempts += attempts[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        # Optimized plotting
        ax.plot(angles, accuracies, linewidth=2, color=self.get_color('teal'), 
                label='Accuracy', marker='o')
        ax.fill(angles, accuracies, color=self.get_color('teal'), alpha=0.25)

        ax.plot(angles, attempts, linewidth=2, linestyle='dashed', 
                color=self.get_color('slate'), label='Attempts', marker='D')
        ax.fill(angles, attempts, color=self.get_color('slate'), alpha=0.20)

        # Optimized styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(subjects, fontweight='bold', fontsize=12)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontweight='bold')
        ax.set_title("Accuracy & Attempts radar", fontsize=16, fontweight='bold', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))


        image_base64 = self.fig_to_base64(fig, backend='matplotlib')  
        return self._standard_response(True, message="Radar chart generated", fig=image_base64)

    def plot_gauge_chart(self, roll_no, test_index=0):
        """Optimized gauge chart with Plotly."""
        student_rank = self.processor.get_student_rank(roll_no, test_index)
        if not student_rank["success"]:
            return self._standard_response(
                False, error=student_rank["error"], message="Failed to get rank data"
            )

        rank_data = student_rank["data"][0]
        score = rank_data["percentage"]
        grade = rank_data["grade"]
        rank_str = f"{rank_data['rank']}/{rank_data['total_students']}"

        # Optimized Plotly gauge
        fig = go.Figure()

        fig.add_trace(go.Indicator(
            mode="gauge",
            value=score,
            gauge={
                'shape': "angular",
                'axis': {'range': [0, 100], 'visible': False},
                'bar': {'color': "#8e44ad", 'thickness': 0.8},
                'bgcolor': "white",
                'borderwidth': 0,
                'steps': [{'range': [0, 100], 'color': "#d3d3d3", 'thickness': 0.5}]
            },
            domain={'x': [0, 1], 'y': [0, 1]},
        ))

        # Batch annotations for better performance
        annotations = [
            dict(x=0.5, y=0.40, showarrow=False, align="center",
                text=f"<span style='font-size:55px; color:#8e44ad; font-family:Arial Black'>{score:.1f}%</span><br>"),
            dict(x=0.5, y=0.95, showarrow=False, text="<b>Overall Performance</b>", font={'size': 24}),
            dict(x=0.5, y=0.20, showarrow=False,
                text=f"<span style='font-size:30px; color:#8e44ad; font-weight:bold'>Grade: {grade}</span><br><br>"
                     f"<span style='font-size:24px; color:#4a4a4a; font-weight:bold'>Rank: {rank_str}</span>")
        ]
        
        fig.update_layout(
            annotations=annotations,
            margin=dict(t=30, b=0, l=20, r=20),
            paper_bgcolor="white",
            height=380,
            width=320
        )

    
        image_base64 = self.fig_to_base64(fig, backend='plotly')  
        return self._standard_response(True, message="Gauge chart created", fig=image_base64)

    def plot_student_comparison(self, roll_no, 
                              subjects=["PHY", "CHEM", "BOT", "ZOO"], 
                              test_index=0):
        """Optimized comparison bar chart."""
        
        subjects = [f"{sub.upper().strip()} TOTAL" for sub in subjects]
        class_averages = self.processor.get_subject_wise_average(subjects, test_index=test_index)
        student_data = self.processor.get_individual_student_data(roll_no, test_index=test_index)

        if not student_data["success"] or not class_averages["success"]:
            return self._standard_response(
                False, error=student_data.get("error"), message="Failed to fetch data"
            )

        student = student_data["data"][0]
        class_avg_dict = class_averages["data"][0]

        # Vectorized score extraction
        student_scores = [student.get(sub, 0) for sub in subjects]
        class_scores = [class_avg_dict.get(sub, 0) for sub in subjects]

        # Optimized bar chart
        x = np.arange(len(subjects))
        width = 0.35
        fig, ax = plt.subplots(figsize=(6, 6))

        bars1 = ax.bar(x - width / 2, student_scores, width, 
                      label='Student', color=self.get_color('blue'))
        bars2 = ax.bar(x + width / 2, class_scores, width, 
                      label='Class Average', color=self.get_color('slate'))

        ax.set_ylabel('Marks', fontsize=12, fontweight='bold')
        ax.set_title('Student vs Class Average', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([sub.split()[0] for sub in subjects], fontsize=12, fontweight='bold')
        ax.legend()

        # Batch text annotations
        for i, (student_score, class_score) in enumerate(zip(student_scores, class_scores)):
            ax.text(x[i] - width / 2, student_score + 1, f'{student_score}', 
                   ha='center', fontsize=10, fontweight='bold')
            ax.text(x[i] + width / 2, class_score + 1, f'{class_score}', 
                   ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
   
        image_base64 = self.fig_to_base64(fig, backend='matplotlib')  
        return self._standard_response(True, message="Comparison bar chart generated", fig=image_base64)
    
    def plot_student_progress_line(self, roll_no, test_index=0):
        """Optimized progress line chart."""
        response = self.processor.get_student_progress(roll_no, test_index=test_index)
        if not response['success']:
            return self._standard_response(
                False, error=response['error'], message="Failed to get student progress"
            )

        progress_data = response['data']
        
        # Vectorized data extraction
        test_indices = [entry['test_index'] for entry in progress_data]
        obtained_marks = [entry['marks'] if entry['status'] == 'present' else None 
                         for entry in progress_data]

        class_avg_response = self.processor.get_class_average()
        if not class_avg_response['success']:
            return self._standard_response(
                False, error=class_avg_response['error'], message="Failed to get class average"
            )

        # Optimized average mapping
        class_avg_dict = {entry['test_index']: entry['class_average'] 
                         for entry in class_avg_response['data']}
        class_averages = [class_avg_dict.get(i, None) for i in test_indices]

        # Optimized plotting
        fig, ax = plt.subplots(figsize=(6, 6))

        ax.plot(test_indices, obtained_marks, linestyle='solid', color=self.get_color('teal'),
                label='Obtained Marks', linewidth=2.5, marker='o', markersize=8)
        ax.plot(test_indices, class_averages, linestyle='dotted', color=self.get_color('orange'),
                label='Class Average', linewidth=2, marker='^', markersize=6)

        # Batch absent annotations
        for i, entry in enumerate(progress_data):
            if entry['status'] == 'absent':
                ax.annotate('Absent', (test_indices[i], 5), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=9, color='red')
        
        
        ax.set_title("Student Progress", fontsize=16, fontweight='bold')
        ax.set_xlabel("Test Index", fontsize=14, fontweight='bold')
        ax.set_ylabel("Marks", fontsize=14, fontweight='bold')
        ax.set_ylim(0, progress_data[0]['total_marks'])
        ax.set_xticks(test_indices)
        ax.set_xticklabels([i + 1 for i in test_indices]) 
        ax.legend(loc='best')

        plt.tight_layout()
   
        image_base64 = self.fig_to_base64(fig, backend='matplotlib')  
        return self._standard_response(True, message="Progress line chart generated", fig=image_base64)

    def clear_cache(self):
        """Clear internal cache to free memory."""
        self._data_cache.clear()
        gc.collect()
        
class ClassVisualizer(DataVisualizerHelper):
    """
    Class for visualizing aggregated class-level data with performance optimizations.
    """
    
    __slots__ = ('_data_cache',)

    def __init__(self, processor):
        """Initialize the ClassVisualizer."""
        super().__init__(processor)
        self._data_cache = {}  # Cache for frequently accessed class data

    
    def _get_class_stats(self, test_index=0, subjects_total=None):
        """Optimized data fetching with caching for class-wide statistics."""
        cache_key = f"stats_{test_index}"
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        response = self.processor.get_marks_distribution_and_subject_stats(test_index=test_index, subjects_total=subjects_total)
        if not response['success']:
            return None
        
        result = response['data'][0]
        self._data_cache[cache_key] = result
        return result

    def plot_score_distribution_bar_chart(self, test_index=0, subjects_total=None):
        """Generates a horizontal bar chart for class score distribution."""
        stats = self._get_class_stats(test_index, subjects_total)
        if not stats:
            return self._standard_response(False, message=f"No stats available for test index {test_index}.")

        dist_data = stats.get('marks_distribution', [])
        if not dist_data:
            return self._standard_response(False, message="No marks distribution data found.")

        df = pd.DataFrame(dist_data)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(df)))

        bars = ax.barh(df['marks_range'], df['student_count'], color=colors, linewidth=1.2)
        
        ax.set_xlabel('Number of Students', fontsize=14, fontweight='bold')
        ax.set_ylabel('Marks Range', fontsize=14, fontweight='bold')
        ax.set_title(f'Score Distribution for Test {test_index + 1}', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.invert_yaxis()

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2., f'{int(width)}', 
                    va='center', ha='left', fontsize=10, fontweight='bold')

        plt.tight_layout()

    
        image_base64 = self.fig_to_base64(fig, backend='matplotlib')  
        return self._standard_response(True, message="Score distribution chart created.", fig=image_base64)


    def plot_subject_accuracy_radar(self, test_index=0, subjects_total=None):
        """Generates a radar chart for class-wide subject accuracy and attempt rates."""

        stats = self._get_class_stats(test_index, subjects_total)
        if not stats:
            return self._standard_response(False, message=f"No stats available for test index {test_index}.")

        subject_stats = stats.get('subject_wise_stats', {})
        if not subject_stats:
            return self._standard_response(False, message="No subject-wise stats found.")

        subjects = list(subject_stats.keys())
        accuracies = []
        attempts = []

        for subj in subjects:
            info = subject_stats[subj]
            acc = info.get("average_accuracy", 0)
            attempt_rate = info.get("average_attempt_rate", 0)
            accuracies.append(acc)
            attempts.append(attempt_rate)

        n_subjects = len(subjects)
        angles = [n / float(n_subjects) * 2 * pi for n in range(n_subjects)]

        # Close the radar
        accuracies += accuracies[:1]
        attempts += attempts[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        # Accuracy plot
        ax.plot(angles, accuracies, linewidth=2, color=self.get_color('teal'), 
                label='Avg Accuracy', marker='o')
        ax.fill(angles, accuracies, color=self.get_color('teal'), alpha=0.25)

        # Attempt rate plot
        ax.plot(angles, attempts, linewidth=2, linestyle='dashed', 
                color=self.get_color('slate'), label='Avg Attempt Rate', marker='D')
        ax.fill(angles, attempts, color=self.get_color('slate'), alpha=0.20)

        # Styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(subjects, fontweight='bold', fontsize=12)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontweight='bold')
        ax.set_title("Class Accuracy & Attempt Rate", fontsize=16, fontweight='bold', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

        
        image_base64 = self.fig_to_base64(fig, backend='matplotlib')  
        return self._standard_response(True, message="Class radar chart created.", fig=image_base64)

    def plot_pass_fail_donut(self, test_index=0):
        """Generates a donut chart for the pass/fail summary."""
        pass_fail_data = self.processor.get_pass_fail_summary( test_index=test_index)
        if not pass_fail_data['success']:
            return self._standard_response(False, message="Could not retrieve pass/fail data.")

        data = pass_fail_data['data']
        passed = data.get('passed_count', 0)
        failed = data.get('failed_count', 0)
        
        if (passed + failed) == 0:
            return self._standard_response(False, message="No students to visualize in pass/fail summary.")

        labels = ['Passed', 'Failed']
        sizes = [passed, failed]
        colors = [self.get_color('green'), self.get_color('red')]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
               wedgeprops=dict(width=0.3, edgecolor='w'), textprops={'fontsize': 14, 'fontweight': 'bold'})

        center_circle = plt.Circle((0,0),0.70,fc='white')
        fig.gca().add_artist(center_circle)
        
        ax.axis('equal')
        ax.set_title(f'Pass/Fail Summary (Pass Mark: 40%)', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        image_base64 = self.fig_to_base64(fig, backend='matplotlib')  
        return self._standard_response(True, message="Pass/fail donut chart created.", fig=image_base64)

class SubjectVisualizer(DataVisualizerHelper):
    def __init__(self, processor):
        super().__init__(processor)

        
    def plot_marks(self, roll_no, test_index=None):
        response = self.processor.get_student_progress_tracking(roll_no, test_index=test_index)
        if not response['success']:
            return self._standard_response(False, error=response['error'], message="Failed to retrieve marks data")

        data = response['data']
        total_marks = data['subject_total_marks_list']
        test_indices = list(range(1, len(total_marks) + 1))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(test_indices, total_marks, linestyle='solid', color=self.get_color('blue'),
                label='Total Marks', linewidth=2.5, marker='s', markersize=8)

        ax.set_title("Subject Total Marks Over Tests", fontsize=16, fontweight='bold')
        ax.set_xlabel("Test Index", fontsize=14, fontweight='bold')
        ax.set_ylabel("Total Marks", fontsize=14, fontweight='bold')
        ax.set_xticks(test_indices)
        ax.set_ylim(0, max(total_marks) + 10)
        ax.legend(loc='best')

        plt.tight_layout()

        
        image_base64 = self.fig_to_base64(fig, backend='matplotlib')  
        return self._standard_response(True, message="Total marks line chart generated", fig=image_base64)
    
    def plot_accuracy(self, roll_no, test_index=None):
        response = self.processor.get_student_progress_tracking(roll_no, test_index=test_index)
        if not response['success']:
            return self._standard_response(False, error=response['error'], message="Failed to retrieve accuracy data")

        data = response['data']
        accuracy_list = data['accuracy_list']
        test_indices = list(range(1, len(accuracy_list) + 1))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(test_indices, accuracy_list, linestyle='solid', color=self.get_color('green'),
                label='Accuracy (%)', linewidth=2.5, marker='o', markersize=8)

        ax.set_title("Student Accuracy Over Tests", fontsize=16, fontweight='bold')
        ax.set_xlabel("Test Index", fontsize=14, fontweight='bold')
        ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight='bold')
        ax.set_xticks(test_indices)
        ax.set_ylim(0, 100)
        ax.legend(loc='best')

        plt.tight_layout()

        
        image_base64 = self.fig_to_base64(fig, backend='matplotlib')  
        return self._standard_response(True, message="Accuracy line chart generated", fig=image_base64)

     