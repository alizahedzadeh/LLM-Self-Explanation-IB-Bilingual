from openai import OpenAI
import re
import json
import logging
import time
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
import pandas as pd

df = pd.read_csv("../../../../datasets/question answering/ai2_arc/ARC-Challenge-clean(persian).csv")

# Configure logging
def setup_logging(log_file: str = "gpt_4o_mini_persian.log"):
    """Setup comprehensive logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{log_file}"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_path}")
    return logger

# Initialize logger
logger = setup_logging()

class Config:
    """Configuration class for the DeepSeek V3 prediction system"""
    # OpenRouter API configuration
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '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')
    MODEL = "openai/gpt-4o-mini"  # DeepSeek V3 model on OpenRouter
    TEMPERATURE = 0.0
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds
    REQUEST_DELAY = 0.5  # delay between requests to avoid rate limiting
    TIMEOUT = 30  # seconds
    MAX_TOKENS = 1000

# Initialize OpenRouter client (compatible with OpenAI client)
client = OpenAI(
    api_key=Config.OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

predict_then_explain_prompt_context = '''
You are given a multiple-choice question written in Persian.

Step 1: Based on your knowledge and reasoning, select the most likely correct answer.  
Step 2: Justify your answer with clear reasoning and explanation.

Instructions:
- Use logical reasoning to determine the best answer.
- Do not reference the other answer options in your explanation.
- Keep the explanation concise but informative (2-4 sentences).
- Provide your explanation in Persian.

---

Question: {question}

Options:
A) {option_A}  
B) {option_B}  
C) {option_C}  
D) {option_D}

Respond in this format:

<prediction>A/B/C/D</prediction>  
<explanation>[توضیح و استدلال به زبان فارسی]</explanation>
'''


def validate_row(row: Dict[str, Any]) -> bool:
    """Validate that a row has all required fields for your data structure"""
    required_fields = ['question_fa', 'choice_A_fa', 'choice_B_fa', 'choice_C_fa', 'choice_D_fa']
    missing_fields = [field for field in required_fields if field not in row or pd.isna(row[field]) or not str(row[field]).strip()]
    
    if missing_fields:
        logger.warning(f"Row missing required fields: {missing_fields}")
        return False
    return True

def extract_prediction_and_explanation(output: str) -> Dict[str, Optional[str]]:
    """Extract prediction and explanation from model output using regex"""
    try:   
        #logger.info(output)
        # Extract prediction
        pred_match = re.search(r"<prediction>\s*([A-D])\s*</prediction>", output, re.IGNORECASE)
        prediction = pred_match.group(1).upper() if pred_match else None
        
        # Extract explanation
        expl_match = re.search(r"<explanation>\s*(.*?)\s*</explanation>", output, re.DOTALL | re.IGNORECASE)
        explanation = expl_match.group(1).strip() if expl_match else None
        
        if not prediction:
            logger.warning("Could not extract prediction from output")
        if not explanation:
            logger.warning("Could not extract explanation from output")
            
        return {
            "prediction": prediction,
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"Error extracting prediction/explanation: {e}")
        return {"prediction": None, "explanation": None}

def make_api_call(prompt: str, row_id: Optional[int] = None) -> Optional[str]:
    """Make API call with retry logic and error handling using OpenRouter client"""
    for attempt in range(Config.MAX_RETRIES):
        try:
            logger.debug(f"Making API call (attempt {attempt + 1}/{Config.MAX_RETRIES}) for row {row_id}")
            
            response = client.chat.completions.create(
                model=Config.MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
                timeout=Config.TIMEOUT,
            )
            
            output = response.choices[0].message.content
            print("MODEL OUTPUT :", output)
            # Log token usage
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                logger.info(f"Row {row_id} - Tokens used: {usage.total_tokens} "
                           f"(prompt: {usage.prompt_tokens}, "
                           f"completion: {usage.completion_tokens})")
            
            return output
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Handle different types of errors
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                wait_time = Config.RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Rate limit hit for row {row_id}, waiting {wait_time}s before retry {attempt + 1}")
                time.sleep(wait_time)
                
            elif "timeout" in error_msg.lower():
                logger.error(f"API timeout for row {row_id}: {error_msg}")
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(Config.RETRY_DELAY)
                    
            elif "api" in error_msg.lower():
                logger.error(f"OpenRouter API error for row {row_id}: {error_msg}")
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(Config.RETRY_DELAY)
                    
            else:
                logger.error(f"Unexpected error ({error_type}) for row {row_id}: {error_msg}")
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(Config.RETRY_DELAY)
    
    logger.error(f"Failed to get response for row {row_id} after {Config.MAX_RETRIES} attempts")
    return None

def predict(row: Dict[str, Any], row_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Enhanced prediction function adapted for your data structure
    
    Args:
        row: Dictionary containing question data with columns: question, choice_A, choice_B, choice_C, choice_D
        row_id: Optional row identifier for logging
        
    Returns:
        Dictionary with prediction results
    """
    start_time = time.time()
    logger.info(f"Starting prediction for row {row_id}, question ID: {row.get('id', 'N/A')}")
    
    # Validate input
    if not validate_row(row):
        logger.error(f"Row {row_id} validation failed")
        return {
            "row_id": row_id,
            "question_id": row.get('id'),
            "prediction": None,
            "explanation": None,
            "raw_output": None,
            "actual_answer": row.get('answerKey'),
            "is_correct": None,
            "error": "Invalid input data",
            "processing_time": 0
        }
    
    try:
        # Format prompt using your column names
        prompt = predict_then_explain_prompt_context.format(
            question=row['question'],
            option_A=row['choice_A'],
            option_B=row['choice_B'],
            option_C=row['choice_C'],
            option_D=row['choice_D'],
        )
        
        logger.debug(f"Row {row_id} - Prompt length: {len(prompt)} characters")
        
        # Make API call
        output = make_api_call(prompt, row_id)
        #logger.info(output)
        if output is None:
            return {
                "row_id": row_id,
                "question_id": row.get('id'),
                "prediction": None,
                "explanation": None,
                "raw_output": None,
                "actual_answer": row.get('answerKey'),
                "is_correct": None,
                "error": "API call failed",
                "processing_time": time.time() - start_time
            }
        
        # Extract prediction and explanation
        extracted = extract_prediction_and_explanation(output)
        
        processing_time = time.time() - start_time
        
        # Check if prediction is correct
        actual_answer = row.get('answerKey')
        predicted_answer = extracted["prediction"]
        is_correct = predicted_answer == actual_answer if (predicted_answer and actual_answer) else None
        
        result = {
            "row_id": row_id,
            "question_id": row.get('id'),
            "prediction": predicted_answer,
            "explanation": extracted["explanation"],
            "raw_output": output,
            "actual_answer": actual_answer,
            "is_correct": is_correct,
            "split": row.get('split'),
            "error": None,
            "processing_time": processing_time
        }
        
        logger.info(f"Row {row_id} completed successfully in {processing_time:.2f}s - "
                   f"Prediction: {result['prediction']}, Actual: {actual_answer}, "
                   f"Correct: {is_correct}")
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Unexpected error processing row {row_id}: {e}")
        return {
            "row_id": row_id,
            "question_id": row.get('id'),
            "prediction": None,
            "explanation": None,
            "raw_output": output,
            "actual_answer": row.get('answerKey'),
            "is_correct": None,
            "error": str(e),
            "processing_time": processing_time
        }

def process_dataframe(df: pd.DataFrame, 
                     save_interval: int = 50,
                     output_file: str = "_predictions_results.json",
                     start_index: int = 0,
                     end_index: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Process a pandas DataFrame with your specific column structure
    
    Args:
        df: DataFrame with columns: question, choice_A, choice_B, choice_C, choice_D, answerKey, etc.
        save_interval: Save results every N processed items
        output_file: File to save results to
        start_index: Start processing from this index (useful for resuming)
        end_index: Stop processing at this index (useful for testing subsets)
        
    Returns:
        List of prediction results
    """
    if end_index is None:
        end_index = len(df)
    
    subset_df = df.iloc[start_index:end_index]
    
    logger.info(f"Starting batch processing of {len(subset_df)} items (rows {start_index} to {end_index-1})")
    logger.info(f"DataFrame info: {len(df)} total rows, columns: {list(df.columns)}")
    
    # Log data distribution
    if 'split' in df.columns:
        split_counts = df['split'].value_counts()
        logger.info(f"Data split distribution: {split_counts.to_dict()}")
    
    if 'answerKey' in df.columns:
        answer_dist = df['answerKey'].value_counts()
        logger.info(f"Answer distribution: {answer_dist.to_dict()}")
    
    results = []
    failed_count = 0
    correct_count = 0
    
    # Create progress bar
    pbar = tqdm(
        subset_df.iterrows(), 
        total=len(subset_df),
        desc="Processing questions with GPT 4o mini",
        unit="question",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    try:
        for original_idx, row in pbar:
            # Process the row
            result = predict(row.to_dict(), row_id=original_idx)
            results.append(result)
            
            # Update counters
            if result['prediction'] is None:
                failed_count += 1
            elif result['is_correct'] is True:
                correct_count += 1
            
            # Update progress bar description
            total_processed = len(results)
            success_rate = ((total_processed - failed_count) / total_processed * 100) if total_processed > 0 else 0
            accuracy = (correct_count / (total_processed - failed_count) * 100) if (total_processed - failed_count) > 0 else 0
            
            pbar.set_postfix({
                'Success': f'{success_rate:.1f}%',
                'Accuracy': f'{accuracy:.1f}%',
                'Last': result['prediction'] or 'FAIL'
            })
            
            # Periodic saving
            if len(results) % save_interval == 0:
                save_results(results, output_file)
                logger.info(f"Intermediate save completed at {len(results)} processed items")
            
            # Rate limiting
            time.sleep(Config.REQUEST_DELAY)
            
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        pbar.close()
        
    except Exception as e:
        logger.error(f"Error during batch processing: {e}")
        pbar.close()
        
    finally:
        # Final save
        save_results(results, output_file)
        
        # Summary statistics
        total_processed = len(results)
        successful = sum(1 for r in results if r['prediction'] is not None)
        correct = sum(1 for r in results if r['is_correct'] is True)
        success_rate = successful / total_processed * 100 if total_processed > 0 else 0
        accuracy = correct / successful * 100 if successful > 0 else 0
        
        logger.info(f"Batch processing completed:")
        logger.info(f"  Total processed: {total_processed}")
        logger.info(f"  Successful predictions: {successful}")
        logger.info(f"  Failed predictions: {total_processed - successful}")
        logger.info(f"  Correct answers: {correct}")
        logger.info(f"  Success rate: {success_rate:.2f}%")
        logger.info(f"  Accuracy rate: {accuracy:.2f}%")
        
        # Split-wise accuracy if available
        if successful > 0:
            splits_accuracy = {}
            for split_name in df['split'].unique() if 'split' in df.columns else ['all']:
                split_results = [r for r in results if r.get('split') == split_name or split_name == 'all']
                split_correct = sum(1 for r in split_results if r['is_correct'] is True)
                split_successful = sum(1 for r in split_results if r['prediction'] is not None)
                if split_successful > 0:
                    splits_accuracy[split_name] = split_correct / split_successful * 100
            
            logger.info(f"Accuracy by split: {splits_accuracy}")
        
    return results

def save_results(results: List[Dict[str, Any]], filename: str):
    """Save results to JSON file with error handling"""
    try:
        output_path = Path("output") 
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        full_path = output_path / f"{timestamp}_{filename}"
        
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Results saved to {full_path}")
        
        # Also save a summary CSV for easy analysis
        summary_path = output_path / f"{timestamp}_deepseek_v3_summary.csv"
        summary_df = pd.DataFrame([
            {
                'row_id': r['row_id'],
                'question_id': r['question_id'],
                'prediction': r['prediction'],
                'actual_answer': r['actual_answer'],
                'is_correct': r['is_correct'],
                'split': r.get('split'),
                'has_error': r['error'] is not None,
                'processing_time': r['processing_time']
            }
            for r in results
        ])
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved to {summary_path}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def analyze_results(results: List[Dict[str, Any]]):
    """Analyze and print results statistics"""
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("GPT 4O mini ANALYSIS RESULTS")
    print("="*50)
    
    # Overall statistics
    total = len(df_results)
    successful = len(df_results[df_results['prediction'].notna()])
    correct = len(df_results[df_results['is_correct'] == True])
    
    print(f"Total questions processed: {total}")
    print(f"Successful predictions: {successful} ({successful/total*100:.1f}%)")
    if successful > 0:
        print(f"Correct predictions: {correct} ({correct/successful*100:.1f}% accuracy)")
    else:
        print("Correct predictions: 0 (No successful predictions)")
    
    # Error analysis
    errors = df_results[df_results['error'].notna()]
    if len(errors) > 0:
        print(f"\nErrors encountered: {len(errors)}")
        error_types = errors['error'].value_counts()
        for error_type, count in error_types.items():
            print(f"  {error_type}: {count}")
    
    # Answer distribution
    if 'actual_answer' in df_results.columns:
        print(f"\nActual answer distribution:")
        print(df_results['actual_answer'].value_counts().sort_index())
    
    if successful > 0 and 'prediction' in df_results.columns:
        print(f"\nPredicted answer distribution:")
        pred_dist = df_results[df_results['prediction'].notna()]['prediction'].value_counts().sort_index()
        print(pred_dist)
    
    # Split analysis if available
    if 'split' in df_results.columns and successful > 0:
        print(f"\nAccuracy by split:")
        for split in df_results['split'].unique():
            if pd.notna(split):
                split_df = df_results[df_results['split'] == split]
                split_correct = len(split_df[split_df['is_correct'] == True])
                split_total = len(split_df[split_df['prediction'].notna()])
                if split_total > 0:
                    accuracy = split_correct / split_total * 100
                    print(f"  {split.upper()}: {split_correct}/{split_total} ({accuracy:.1f}%)")

# Test API connection
def test_api_connection():
    """Test if the OpenRouter API is working properly with Gpt 4o mini"""
    try:
        logger.info("Testing OpenRouter Gpt 4o mini API connection...")
        
        response = client.chat.completions.create(
            model=Config.MODEL,
            messages=[{"role": "user", "content": "Hello, please respond with 'DeepSeek V3 API working'."}],
            max_tokens=10,
            temperature=0,
            extra_headers={
                "HTTP-Referer": "https://github.com/alizahedzadeh",
                "X-Title": "Gpt 4o mini API Test"
            }
        )
        
        result = response.choices[0].message.content
        logger.info(f"API test successful. Response: {result}")
        return True
        
    except Exception as e:
        logger.error(f"API test failed: {e}")
        return False


# Example usage
if __name__ == "__main__":
    # Set your OpenRouter API key
    # os.environ['OPENROUTER_API_KEY'] = 'your_openrouter_api_key_here'
    
    # Test API connection first
    if not test_api_connection():
        print("❌ API connection failed. Please check your OpenRouter API key and internet connection.")
        exit(1)
    
    # Load your data
    base_df = df.copy()
    base_df.head()
    base_df.dropna(inplace=True)
    base_df.info()
    base_df['answerKey'].value_counts()
    mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
    base_df['answerKey'] = base_df['answerKey'].replace(mapping)
    base_df = base_df[base_df['answerKey'].isin(['A', 'B', 'C', 'D'])]
    base_df.info()

    # Configuration for full dataset processing
    Config.REQUEST_DELAY = 0.3  # Conservative rate limiting for OpenRouter
    Config.MAX_RETRIES = 5      # More retries for stability
    Config.TIMEOUT = 45         # Longer timeout for complex questions

    # Log the start of full processing
    logger.info(f"Starting full dataset processing with DeepSeek V3 by user: alizahedzadeh")
    logger.info(f"Dataset size: {len(base_df)} questions")
    logger.info(f"Processing start time (UTC): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Process the full dataset
    print("🚀 Starting full dataset processing with DeepSeek V3...")
    print(f"📊 Total questions: {len(base_df)}")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Process in manageable chunks to handle potential interruptions
    chunk_size = 500
    total_chunks = (len(base_df) + chunk_size - 1) // chunk_size

    print(f"📦 Processing in {total_chunks} chunks of {chunk_size} questions each")

    all_results = []

    for chunk_num in range(total_chunks):
        start_idx = chunk_num * chunk_size
        end_idx = min(start_idx + chunk_size, len(base_df))
        
        print(f"\n{'='*60}")
        print(f"🔄 Processing Chunk {chunk_num + 1}/{total_chunks} with DeepSeek V3")
        print(f"📍 Rows {start_idx} to {end_idx-1} ({end_idx - start_idx} questions)")
        print(f"{'='*60}")
        
        chunk_results = process_dataframe(
            base_df, 
            start_index=start_idx, 
            end_index=end_idx,
            save_interval=50,  # Save every 50 questions
            output_file=f"gpt_4o_mini_chunk_{chunk_num + 1}_predictions.json"
        )
        
        all_results.extend(chunk_results)
        
        # Log chunk completion
        chunk_successful = sum(1 for r in chunk_results if r['prediction'] is not None)
        chunk_accuracy = sum(1 for r in chunk_results if r['is_correct'] is True) / chunk_successful * 100 if chunk_successful > 0 else 0
        
        print(f"✅ Chunk {chunk_num + 1} completed:")
        print(f"   📈 Success rate: {chunk_successful}/{len(chunk_results)} ({chunk_successful/len(chunk_results)*100:.1f}%)")
        print(f"   🎯 Accuracy: {chunk_accuracy:.1f}%")
        
        # Small break between chunks
        if chunk_num < total_chunks - 1:
            print("⏸️  Brief pause between chunks...")
            time.sleep(2)

    # Save final combined results
    print(f"\n{'='*60}")
    print("💾 Saving final combined results...")

    final_output_file = f"gpt_4o_mini_full_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(all_results, final_output_file)

    # Final analysis
    print(f"\n{'='*60}")
    print("📊 FINAL ANALYSIS - FULL DATASET WITH Gpt 4o mini")
    print(f"{'='*60}")

    analyze_results(all_results)

    # Additional detailed statistics
    df_results = pd.DataFrame(all_results)

    # Performance metrics
    total_time = sum(r['processing_time'] for r in all_results)
    avg_time_per_question = total_time / len(all_results)

    print(f"\n⏱️  PERFORMANCE METRICS:")
    print(f"   Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   Average time per question: {avg_time_per_question:.2f} seconds")

    # Token usage summary (if available in logs)
    print(f"\n💰 COST ESTIMATION:")
    print(f"   Total questions processed: {len(all_results)}")
    print(f"   Model: Gpt 4o mini")
    print(f"   Estimated tokens per question: ~800-1200")
    print(f"   Estimated total tokens: ~{len(all_results) * 1000:,}")

    # Final success message
    final_successful = sum(1 for r in all_results if r['prediction'] is not None)
    final_accuracy = sum(1 for r in all_results if r['is_correct'] is True) / final_successful * 100 if final_successful > 0 else 0

    print(f"\n🏆 FINAL RESULTS SUMMARY:")
    print(f"   📊 Total Questions: {len(all_results)}")
    print(f"   ✅ Successful Predictions: {final_successful} ({final_successful/len(all_results)*100:.1f}%)")
    print(f"   🎯 Overall Accuracy: {final_accuracy:.1f}%")

    logger.info(f"Full dataset processing completed with Gpt 4o mini by alizahedzadeh. Success rate: {final_successful/len(all_results)*100:.2f}%, Accuracy: {final_accuracy:.2f}%")