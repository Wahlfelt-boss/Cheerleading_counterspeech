import csv
import sys
import requests
import json
import os
import time
import traceback
import argparse # Added for command-line arguments

# --- API Key Check ---
API_KEY = os.environ.get('GEMINI_API_KEY')
if not API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.")
    # Add detailed instructions as before
    print("Please set the environment variable before running the script.")
    print("Example (Linux/macOS): export GEMINI_API_KEY='YOUR_KEY_HERE'")
    print("Example (Windows CMD): set GEMINI_API_KEY=YOUR_KEY_HERE")
    print("Example (Windows PowerShell): $env:GEMINI_API_KEY='YOUR_KEY_HERE'")
    sys.exit(1)

# --- MODIFIED: Classification Function accepts model_name ---
def classify_with_gemini(hatespeech: str, response_text: str, model_name: str) -> tuple[int, str]:
    """
    Classifies the response_text as COUNTERSPEECH or NORMAL,
    considering the context of the original hatespeech.
    Uses the specified Gemini model via REST API.
    Returns: Tuple (result_code, reason_string)
    """
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"
    headers = {"Content-Type": "application/json"}

    prompt = f"""Given the following HATESPEECH and the RESPONSE text, classify the RESPONSE text as either 'COUNTERSPEECH' or 'NORMAL'.
A COUNTERSPEECH response directly addresses or counters the hate speech.
A NORMAL response might be unrelated, neutral, or even agreeing with the hate speech.
Respond with only one word: COUNTERSPEECH or NORMAL.

HATESPEECH: "{hatespeech}"

RESPONSE: "{response_text}"

Classification:"""

    request_body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "safetySettings": [ # Using relaxed settings based on previous success
             { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH" },
             { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH" },
             { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH" },
             { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH" },
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 3000,
            "topP": 0.95,
            "topK": 40
        }
    }

    hatespeech_log = hatespeech[:50] + ('...' if len(hatespeech) > 50 else '')
    response_log = response_text[:50] + ('...' if len(response_text) > 50 else '')
    # Use the passed model_name in the print statement
    print(f"Sending to {model_name}: Response to '{hatespeech_log}' -> '{response_log}'")

    try:
        response = requests.post(api_url, headers=headers, json=request_body, timeout=90)
        response.raise_for_status()
        response_json = response.json()

        if 'candidates' in response_json and response_json['candidates']:
            candidate = response_json['candidates'][0]
            finish_reason = candidate.get('finishReason', 'UNKNOWN')

            if finish_reason not in ['STOP', 'MAX_TOKENS']:
                 if finish_reason == 'SAFETY':
                     safety_ratings = candidate.get('safetyRatings', [])
                     block_reasons = [f"{rating.get('category')}={rating.get('probability')}" for rating in safety_ratings if rating.get('blocked')]
                     reason = f"Blocked by API Safety Filter ({finish_reason}): {', '.join(block_reasons) if block_reasons else 'No specific category given'}"
                     print(f"Warning: {reason}")
                     return -1, reason
                 reason = f"API call finished unexpectedly: {finish_reason}"
                 print(f"Warning: {reason}")
                 if 'content' not in candidate or 'parts' not in candidate['content'] or not candidate['content']['parts']:
                     reason += f" | Candidate: {json.dumps(candidate)}"
                     return -1, reason

            if 'content' in candidate and 'parts' in candidate['content'] and candidate['content']['parts']:
                predicted_label_str = candidate['content']['parts'][0].get('text', '').strip().upper()
                raw_text_log = predicted_label_str[:30] + ('...' if len(predicted_label_str) > 30 else '')
                print(f"  -> API Response Raw Text: '{raw_text_log}'")

                is_counter = 'COUNTERSPEECH' in predicted_label_str
                is_normal = 'NORMAL' in predicted_label_str

                if is_counter and not is_normal: return 1, f"OK (Detected COUNTERSPEECH in '{raw_text_log}')"
                elif is_normal and not is_counter: return 0, f"OK (Detected NORMAL in '{raw_text_log}')"
                elif predicted_label_str == 'COUNTERSPEECH': return 1, "OK"
                elif predicted_label_str == 'NORMAL': return 0, "OK"
                else:
                    reason = f"Unexpected Model Response Format: '{raw_text_log}' | FinishReason: {finish_reason} | Candidate: {json.dumps(candidate)}"
                    print(f"Warning: Unexpected Model Response Format: '{raw_text_log}' (FinishReason: {finish_reason})")
                    return -1, reason
            else:
                 reason = f"Missing content/parts after FinishReason={finish_reason} | Candidate: {json.dumps(candidate)}"
                 print(f"Warning: {reason}")
                 return -1, reason

        elif 'promptFeedback' in response_json and 'blockReason' in response_json['promptFeedback']:
             block_reason = response_json['promptFeedback']['blockReason']
             details = response_json['promptFeedback'].get('blockReasonMessage', 'No details provided')
             reason = f"Prompt Blocked by API ({block_reason}): {details}"
             print(f"Warning: {reason}")
             return -1, reason
        else:
            reason = f"Unexpected API Response Structure: {json.dumps(response_json)}"
            print(f"Warning: {reason}")
            return -1, reason

    except requests.exceptions.Timeout:
        reason = "Network Timeout during API call"
        print(f"Error: {reason} for response: '{response_log}'")
        return -1, reason
    except requests.exceptions.RequestException as req_e:
        status_code = getattr(req_e.response, 'status_code', 'N/A')
        reason = f"Network/HTTP Error during API call (Status: {status_code}): {req_e}"
        print(f"Error: {reason} for response: '{response_log}'")
        return -1, reason
    except Exception as e:
        reason = f"Unexpected Error during API call processing: {e.__class__.__name__}: {e}"
        print(f"Error: {reason} for response: '{response_log}'")
        traceback.print_exc()
        return -1, reason

# --- MODIFIED: Evaluation function accepts model_name ---
def evaluate_classifier(dataset_path: str, log_file_path: str, model_name: str):
    """
    Reads dataset, classifies responses using the specified model, evaluates accuracy,
    logs issues, and skips rows already present in the log file.
    """
    processed_row_numbers = set()
    # --- Load processed rows (logic remains the same) ---
    if os.path.exists(log_file_path):
        try:
            print(f"Found existing log file: {log_file_path}. Reading processed rows...")
            with open(log_file_path, 'r', encoding='utf-8', newline='') as log_read_file:
                log_reader = csv.reader(log_read_file)
                header = next(log_reader, None) # Read header

                if header and (not header or header[0].strip().lower() != 'rownumber'):
                    print(f"Warning: Log file header mismatch or empty. Expected 'RowNumber' as first column, got '{header[0] if header else 'None'}'. Attempting to proceed.")

                processed_count = 0
                malformed_count = 0
                for i, log_row in enumerate(log_reader):
                    if log_row and len(log_row) > 0:
                        try:
                            row_num = int(log_row[0])
                            processed_row_numbers.add(row_num)
                            processed_count += 1
                        except (ValueError, TypeError):
                            if malformed_count < 5:
                                print(f"Warning: Skipping invalid row number format in log file at line {i+2}: '{log_row[0]}'")
                            malformed_count += 1
                            continue
                if malformed_count > 5:
                    print(f"... (suppressed {malformed_count - 5} further malformed row number warnings)")
            print(f"Loaded {processed_count} previously processed row numbers (encountered {malformed_count} malformed log entries).")
        except FileNotFoundError:
            print(f"Log file {log_file_path} not found. Starting fresh.")
            processed_row_numbers = set()
        except StopIteration:
             print("Log file is empty or contains only a header. Starting fresh.")
             processed_row_numbers = set()
        except Exception as e:
            print(f"Error reading existing log file {log_file_path}: {e.__class__.__name__}: {e}. Proceeding without skipping rows.")
            processed_row_numbers = set()

    # --- Counters for the current run ---
    current_run_rows_visited = 0
    current_run_skipped_invalid_data = 0
    current_run_api_attempts = 0
    current_run_api_errors = 0
    current_run_successful_classifications = 0
    current_run_correct_predictions = 0
    current_run_skipped_already_processed = 0

    log_headers = ['RowNumber', 'Hatespeech', 'ResponseText', 'TrueLabel', 'PredictedLabel', 'Status', 'Reason']
    needs_header = not os.path.exists(log_file_path) or (os.path.exists(log_file_path) and os.path.getsize(log_file_path) == 0)
    log_writer = None

    try:
        # --- File Handling (logic remains the same) ---
        with open(dataset_path, 'r', encoding='utf-8') as infile, \
             open(log_file_path, 'a', encoding='utf-8', newline='') as logfile:

            reader = csv.DictReader(infile)
            log_writer = csv.writer(logfile)

            if needs_header:
                log_writer.writerow(log_headers)
                print(f"Created or writing header to log file: {log_file_path}")
                logfile.flush()

            # --- Check Dataset Columns (logic remains the same) ---
            required_columns = ['hatespeech', 'response text', 'label']
            if not all(col in reader.fieldnames for col in required_columns):
                 missing = [col for col in required_columns if col not in reader.fieldnames]
                 error_message = f"Dataset file '{dataset_path}' must contain columns: {', '.join(required_columns)}. Missing: {', '.join(missing)}"
                 print(f"Error: {error_message}")
                 if log_writer:
                      try:
                           log_writer.writerow(['N/A', '', '', '', '', 'Critical Error', error_message])
                           logfile.flush()
                      except Exception as log_err:
                           print(f"Additionally, failed to write critical error to log file: {log_err}")
                 sys.exit(1)

            print(f"Starting/Resuming evaluation using dataset: {dataset_path}")
            print(f"Using model: {model_name}") # Print the model being used
            print(f"Logging issues to: {log_file_path}")

            # --- Process Dataset Rows ---
            for i, row in enumerate(reader):
                row_num = i + 2
                current_run_rows_visited += 1

                # --- Check if already processed (logic remains the same) ---
                if row_num in processed_row_numbers:
                    if current_run_skipped_already_processed < 5:
                         print(f"Skipping row {row_num} (already processed).")
                    elif current_run_skipped_already_processed == 5:
                         print("...(suppressing further 'already processed' messages)...")
                    current_run_skipped_already_processed += 1
                    continue

                # --- Process This Row (logic remains the same) ---
                hatespeech_text = row.get('hatespeech', '').strip()
                response_text = row.get('response text', '').strip()
                label_str = row.get('label', '').strip()
                true_label = -99

                hatespeech_log_entry = hatespeech_text[:500] + ('...' if len(hatespeech_text) > 500 else '')
                response_log_entry = response_text[:500] + ('...' if len(response_text) > 500 else '')
                log_entry_base = [row_num, hatespeech_log_entry, response_log_entry, label_str]
                skip_reason = None

                if not hatespeech_text: skip_reason = "Empty hatespeech"
                elif not response_text: skip_reason = "Empty response text"
                else:
                    try:
                        true_label = int(label_str)
                        if true_label not in [0, 1]:
                            skip_reason = f"Invalid label value '{label_str}'. Expected 0 or 1."
                            true_label = -99
                    except (ValueError, TypeError):
                        skip_reason = f"Invalid/Missing label '{label_str}'. Expected integer 0 or 1."
                        true_label = -99

                if skip_reason:
                    print(f"Warning: Skipping row {row_num} due to {skip_reason}.")
                    log_writer.writerow(log_entry_base + ['', 'Skipped', skip_reason])
                    logfile.flush()
                    current_run_skipped_invalid_data += 1
                    continue

                # --- MODIFIED: Pass model_name to classifier ---
                current_run_api_attempts += 1
                predicted_label_code, reason = classify_with_gemini(hatespeech_text, response_text, model_name)

                # --- Log results (logic remains the same) ---
                if predicted_label_code == -1:
                    current_run_api_errors += 1
                    log_writer.writerow(log_entry_base[:4] + [predicted_label_code, 'API Error', reason[:2000]])
                else:
                    current_run_successful_classifications += 1
                    status = 'Correct'
                    if predicted_label_code != true_label:
                        status = 'Misclassified'
                        print(f"Info: Misclassification on row {row_num}. True: {true_label}, Predicted: {predicted_label_code}")
                    log_writer.writerow(log_entry_base[:4] + [predicted_label_code, status, reason[:1000]])
                    if status == 'Correct':
                        current_run_correct_predictions += 1

                logfile.flush()

                # --- Progress printout (logic remains the same) ---
                if current_run_api_attempts > 0 and current_run_api_attempts % 10 == 0:
                   print(f"Rows visited this run: {current_run_rows_visited}, API Attempts this run: {current_run_api_attempts}...")
                   # time.sleep(0.5)

    # --- Exception Handling (logic remains the same) ---
    except FileNotFoundError:
        error_message = f"Dataset file not found at '{dataset_path}'"
        print(f"Error: {error_message}")
        if log_writer:
             try:
                  log_writer.writerow(['N/A', '', '', '', '', 'Critical Error', error_message])
                  if 'logfile' in locals() and logfile and not logfile.closed: logfile.flush()
             except Exception as log_err: print(f"Additionally, failed to write critical error to log file: {log_err}")
        sys.exit(1)
    except Exception as e:
        error_message = f"An unexpected error occurred during evaluation: {e.__class__.__name__}: {e}"
        print(f"Error: {error_message}")
        traceback.print_exc()
        if log_writer:
             try:
                  log_writer.writerow(['N/A', '', '', '', '', 'Critical Error', error_message])
                  if 'logfile' in locals() and logfile and not logfile.closed: logfile.flush()
             except Exception as log_err: print(f"Additionally, failed to write critical error to log file: {log_err}")
        sys.exit(1)

    # --- Final Summary Output (logic remains the same) ---
    print("\n--- Evaluation Summary (Current Run) ---")
    print(f"Total dataset rows visited in this run: {current_run_rows_visited}")
    print(f"Rows skipped (already processed in previous runs): {current_run_skipped_already_processed}")
    print(f"Rows skipped (invalid data in this run): {current_run_skipped_invalid_data}")
    print(f"API classification attempts in this run: {current_run_api_attempts}")
    print(f"API errors in this run: {current_run_api_errors}")
    print(f"Successful classifications in this run: {current_run_successful_classifications}")
    print(f"Correct predictions in this run: {current_run_correct_predictions}")

    if current_run_successful_classifications > 0:
        accuracy = (current_run_correct_predictions / current_run_successful_classifications) * 100
        print(f"Accuracy (Correct / Successful Classifications in this run): {accuracy:.2f}%")
    elif current_run_api_attempts > 0:
        print("Accuracy (this run): N/A (No classifications were successfully completed)")
    else:
        print("Accuracy (this run): N/A (No new valid rows required API classification attempt)")

    print(f"\nCumulative results (including previous runs) are stored in: {log_file_path}")


# --- MODIFIED: Main Execution Block with argparse ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate counter-speech classification using a Gemini model.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name of the Gemini model to use (e.g., 'gemini-1.5-flash-latest', 'gemini-1.5-pro-latest')."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='1000_rows.csv',
        help="Path to the input dataset CSV file (default: 1000_rows.csv)."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None, # Default will be generated based on model name
        help="Path to the output log CSV file (default: classification_log_MODEL_NAME.csv)."
    )

    args = parser.parse_args()

    # Generate default log file name if not provided
    log_file_name = args.log_file
    if log_file_name is None:
        # Sanitize model name for use in filename (replace slashes, etc.)
        safe_model_name = args.model.replace('/', '_').replace(':', '_')
        log_file_name = f'classification_log_{safe_model_name}.csv'

    # Call the evaluation function with parsed arguments
    evaluate_classifier(args.dataset, log_file_name, args.model)
