import csv
import os
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import numpy as np # Import numpy

def analyze_classification_log(filename="classification_log.csv"):
    """
    Analyzes a classification log file (CSV format) to compute evaluation metrics.

    Args:
        filename (str): The path to the CSV classification log file.
                        Expected columns: TrueLabel, PredictedLabel.

    Returns:
        dict: A dictionary containing the analysis results or an error message.
              Keys: 'filename', 'confusion_matrix_str', 'f1_score', 'classification_report', 'error'
    """
    true_labels = []
    predicted_labels = []
    unique_labels_overall = set()

    try:
        with open(filename, 'r', newline='') as csvfile:
            # Automatically detect header and skip it
            try:
                dialect = csv.Sniffer().sniff(csvfile.read(1024))
                csvfile.seek(0)
                reader = csv.reader(csvfile, dialect)
                header = next(reader) # Read header
            except csv.Error:
                csvfile.seek(0)
                reader = csv.reader(csvfile)
                try:
                    header = next(reader)
                except StopIteration:
                     return {'filename': filename, 'error': "CSV file is empty."}
            except StopIteration: # Handle empty file after header read attempt
                 return {'filename': filename, 'error': "CSV file is empty or contains only a header."}


            # Find indices for 'TrueLabel' and 'PredictedLabel', case-insensitive
            try:
                header_lower = [h.strip().lower() for h in header]
                true_label_idx = header_lower.index('truelabel')
                predicted_label_idx = header_lower.index('predictedlabel')
            except ValueError:
                 return {'filename': filename, 'error': "CSV must contain 'TrueLabel' and 'PredictedLabel' columns."}


            for row in reader:
                 if len(row) > max(true_label_idx, predicted_label_idx):
                     try:
                         true_label = int(row[true_label_idx].strip())
                         predicted_label = int(row[predicted_label_idx].strip())
                         true_labels.append(true_label)
                         predicted_labels.append(predicted_label)
                         unique_labels_overall.add(true_label)
                         unique_labels_overall.add(predicted_label)
                     except (ValueError, IndexError):
                             pass # Silently skip rows with format/conversion errors


            if not true_labels: # Check if list is empty after processing
                return {'filename': filename, 'error': "No valid classification data found to analyze."}

            # ---- TN/FP/FN/TP Calculation (Focus on labels 0 and 1) ----
            # Filter data for binary (0 vs 1) analysis if necessary
            binary_true = []
            binary_pred = []
            labels_present_binary = set()
            for t, p in zip(true_labels, predicted_labels):
                 # Include if either true or predicted is 0 or 1
                 # This ensures we account for misclassifications involving 0 or 1
                 if t in [0, 1] or p in [0, 1]:
                     binary_true.append(t)
                     binary_pred.append(p)
                     labels_present_binary.add(t)
                     labels_present_binary.add(p)

            # Calculate confusion matrix for labels 0 and 1 specifically
            # We use labels=[0, 1] to force a 2x2 matrix structure.
            # This handles cases where only 0s or only 1s might be present in the filtered set.
            if 0 in labels_present_binary or 1 in labels_present_binary:
                 # Use numpy arrays for indexing compatibility with sklearn labels
                 cm_binary = confusion_matrix(np.array(binary_true), np.array(binary_pred), labels=[0, 1])
                 tn, fp, fn, tp = cm_binary.ravel()
                 cm_str = f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
            else:
                 # Handle case where neither 0 nor 1 are present at all
                 # This case might be unlikely if the primary goal involves 0/1, but handles edge cases.
                 # TN, FP, FN, TP are all zero in this context for labels 0 and 1.
                 cm_str = "TN=0, FP=0, FN=0, TP=0" # Or indicate not applicable: "N/A for labels 0/1"

            # ---- Overall Metrics Calculation (Using all original labels) ----
            # Calculate F1 score using all original labels
            f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

            # Get classification report as string using all original labels
            sorted_unique_labels = sorted(list(unique_labels_overall))
            # Ensure target names are strings for the report
            target_names = [str(l) for l in sorted_unique_labels]
            # Use labels= parameter to ensure consistent report structure across files
            report_str = classification_report(true_labels, predicted_labels, labels=sorted_unique_labels, target_names=target_names, zero_division=0)


            return {
                'filename': filename,
                'confusion_matrix_str': cm_str, # Use the TN/FP/FN/TP string
                'f1_score': f1,
                'classification_report': report_str
            }

    except FileNotFoundError:
        return {'filename': filename, 'error': f"File not found: {filename}"}
    except Exception as e:
        # More specific error logging can be added here if needed
        return {'filename': filename, 'error': f"An unexpected error occurred: {e}"}


def run_analysis_on_logs(log_patterns=["classification_log*.csv"], output_file="classification_analysis_summary.txt"):
    """
    Runs the analysis on all log files matching the patterns and writes a summary.

    Args:
        log_patterns (list): A list of filename patterns to match (e.g., ["log_*.csv"]).
        output_file (str): The file to write the consolidated analysis summary.
    """
    import glob # Use glob directly inside the function
    import sys # For stderr

    all_results = []
    log_files = []
    for pattern in log_patterns:
        found_files = glob.glob(pattern)
        if not found_files:
             print(f"Warning: No files found matching pattern '{pattern}'", file=sys.stderr)
        log_files.extend(found_files) # Find files matching the pattern

    if not log_files:
        print("Error: No log files found matching any of the patterns.", file=sys.stderr)
        return

    # Sort files for consistent output order
    log_files.sort()


    for log_file in log_files:
        print(f"Analyzing {log_file}...")
        results = analyze_classification_log(log_file)
        all_results.append(results)

    # Write summary to output file
    try:
        with open(output_file, 'w') as f:
            f.write("Classification Analysis Summary") # Added newline
            f.write("===============================") # Added newline
            for result in all_results:
                f.write(f"--- Analysis for: {result.get('filename', 'N/A')} ---") # Added newline
                if 'error' in result:
                    f.write(f"Error: {result['error']}") # Added newline
                else:
                    # Use the specific key for the TN/FP/FN/TP string
                    f.write(f"Confusion Matrix (Binary 0 vs 1): {result.get('confusion_matrix_str', 'N/A')}") # Added newline and clarification
                    f.write(f"F1 Score (Macro Average): {result.get('f1_score', 'N/A'):.4f}") # Format F1 score, added newline
                    f.write("Classification Report:") # Added newline
                    f.write(f"{result.get('classification_report', 'N/A')}") # Added newline
                f.write("-" * 50 + "") # Separator, added newline
        print(f"Analysis summary written to {output_file}")
    except IOError as e:
        print(f"Error writing summary file '{output_file}': {e}", file=sys.stderr)


# Example usage:
if __name__ == "__main__":
     # Analyze specific files or patterns
     run_analysis_on_logs(log_patterns=["classification_log*.csv"], output_file="classification_analysis_summary.txt")
     # Example analyzing a single file:
     # run_analysis_on_logs(log_patterns=["classification_log_gemini-2.5-pro-preview-03-25.csv"], output_file="pro_analysis.txt")
