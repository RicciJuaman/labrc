"""
Automated relevance labeling for RAG evaluation using OpenAI API.
Processes CSV with reviews and classifies relevance for each query.

Requirements:
    pip install openai pandas python-dotenv tqdm

Usage:
    python label_relevance.py --input reviews.csv --output labeled_reviews.csv
"""

import os
import pandas as pd
from openai import OpenAI
from typing import Dict, List
from dotenv import load_dotenv
from tqdm import tqdm
import json
import time
import argparse

load_dotenv()


class RelevanceLabeler:
    """Classify document relevance using OpenAI API."""
    
    # Define your queries here
    QUERIES = {
        "Good Dog Food": "Reviews mentioning positive experiences with dog food quality, ingredients, or general satisfaction",
        "Bad dog food": "Reviews mentioning negative experiences, poor quality, or dissatisfaction with dog food",
        "dog got sick": "Reviews mentioning health issues, illness, vomiting, diarrhea, or adverse reactions after eating",
        "dog is picky eater": "Reviews mentioning dogs being selective, refusing food, or being finicky about eating",
        "delivery issue": "Reviews mentioning shipping problems, damaged packages, late delivery, or logistics issues",
        "grain free dog food": "Reviews specifically mentioning grain-free formulas or ingredients",
        "refund": "Reviews mentioning returns, refunds, money back, or getting reimbursed"
    }
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini", batch_mode: bool = False):
        """
        Initialize the labeler.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: OpenAI model to use (gpt-4o-mini is cheap and fast, gpt-4o for higher quality)
            batch_mode: Use OpenAI Batch API for 50% discount (slower, 24h turnaround)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.batch_mode = batch_mode
        print(f"Using model: {self.model}")
        if batch_mode:
            print("⚡ Batch mode enabled - 50% cheaper, 24h processing time")
        
    def create_prompt(self, summary: str, text: str, queries: Dict[str, str]) -> str:
        """Create prompt for GPT to classify relevance."""
        
        # Build query list
        query_list = ""
        for i, (query_name, description) in enumerate(queries.items(), 1):
            query_list += f"{i}. **{query_name}**: {description}\n"
        
        prompt = f"""You are a relevance classifier for a product review search system.

Given a dog food review, determine if it's relevant to each query below.

**Review Summary:** {summary or "N/A"}

**Review Text:** {text[:1000]}...

**Queries to classify:**
{query_list}

For each query, determine if this review is relevant:
- **YES** = The review clearly discusses this topic (even briefly)
- **NO** = The review does not mention or relate to this topic

**IMPORTANT:** 
- Be strict but fair - a review must actually mention the topic
- "Good Dog Food" should match ANY positive sentiment
- "Bad dog food" should match ANY negative sentiment
- For specific topics (sick, picky, delivery, grain-free, refund), require explicit mention

Respond ONLY with a JSON object in this EXACT format (no markdown, no extra text):
{{
  "Good Dog Food": "YES",
  "Bad dog food": "NO",
  "dog got sick": "NO",
  "dog is picky eater": "YES",
  "delivery issue": "NO",
  "grain free dog food": "NO",
  "refund": "NO"
}}"""
        
        return prompt
    
    def classify_review(self, summary: str, text: str) -> Dict[str, str]:
        """
        Classify a single review for all queries.
        
        Returns:
            Dict mapping query name to "YES" or "NO"
        """
        prompt = self.create_prompt(summary, text, self.QUERIES)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise relevance classifier. Always respond with valid JSON only."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0,  # Deterministic for consistency
                    max_tokens=500,
                    response_format={"type": "json_object"}  # Force JSON output
                )
                
                # Extract response
                response_text = response.choices[0].message.content.strip()
                
                # Parse JSON
                labels = json.loads(response_text)
                
                # Validate all queries are present
                if not all(q in labels for q in self.QUERIES.keys()):
                    raise ValueError("Missing queries in response")
                
                return labels
                
            except json.JSONDecodeError as e:
                print(f"JSON parse error (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Response: {response_text}")
                if attempt == max_retries - 1:
                    # Return all NO as fallback
                    return {q: "NO" for q in self.QUERIES.keys()}
                time.sleep(1)
                
            except Exception as e:
                print(f"API error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return {q: "NO" for q in self.QUERIES.keys()}
                time.sleep(2)
        
        return {q: "NO" for q in self.QUERIES.keys()}
    
    def process_csv(
        self, 
        input_path: str, 
        output_path: str,
        start_row: int = 0,
        max_rows: int = None,
        checkpoint_interval: int = 50
    ):
        """
        Process CSV file and add relevance labels.
        
        Args:
            input_path: Path to input CSV
            output_path: Path to save labeled CSV
            start_row: Row to start from (for resuming)
            max_rows: Maximum rows to process (None = all)
            checkpoint_interval: Save progress every N rows
        """
        # Load CSV
        print(f"Loading {input_path}...")
        df = pd.read_csv(input_path)
        
        print(f"Found {len(df)} reviews")
        print(f"Columns: {df.columns.tolist()}")
        
        # Validate required columns
        required = ['Id', 'Summary', 'Text']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Add query columns if they don't exist
        for query in self.QUERIES.keys():
            if query not in df.columns:
                df[query] = ""
        
        # Determine rows to process
        end_row = min(len(df), start_row + max_rows) if max_rows else len(df)
        rows_to_process = range(start_row, end_row)
        
        print(f"\nProcessing rows {start_row} to {end_row}")
        print(f"Queries: {list(self.QUERIES.keys())}")
        print(f"Using model: {self.model}")
        print(f"Checkpoint interval: {checkpoint_interval} rows\n")
        
        # Process each row
        processed_count = 0
        error_count = 0
        
        for idx in tqdm(rows_to_process, desc="Labeling reviews"):
            row = df.iloc[idx]
            
            # Skip if already labeled (all queries have YES or NO)
            already_labeled = all(
                str(row.get(q, "")).strip() in ["YES", "NO"]
                for q in self.QUERIES.keys()
            )
            
            if already_labeled:
                continue
            
            try:
                # Classify
                summary = str(row.get('Summary', ''))
                text = str(row.get('Text', ''))
                
                if not text or text == 'nan':
                    # No text, mark all as NO
                    labels = {q: "NO" for q in self.QUERIES.keys()}
                else:
                    labels = self.classify_review(summary, text)
                
                # Update dataframe
                for query, label in labels.items():
                    df.at[idx, query] = label
                
                processed_count += 1
                
                # Checkpoint save
                if processed_count % checkpoint_interval == 0:
                    df.to_csv(output_path, index=False)
                    print(f"\n✓ Checkpoint: Saved {processed_count} rows to {output_path}")
                
                # Rate limiting (optional for OpenAI, but good practice)
                # gpt-4o-mini has high rate limits (10k RPM, 200M TPM)
                # Only add delay if hitting rate limits
                if processed_count % 100 == 0 and processed_count > 0:
                    time.sleep(1)  # Brief pause every 100 requests
                
            except Exception as e:
                print(f"\n✗ Error processing row {idx} (ID: {row.get('Id')}): {e}")
                error_count += 1
                
                # Fill with NO on error
                for query in self.QUERIES.keys():
                    if not df.at[idx, query] or df.at[idx, query] == '':
                        df.at[idx, query] = "NO"
        
        # Final save
        df.to_csv(output_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"✓ Processing complete!")
        print(f"  Processed: {processed_count} reviews")
        print(f"  Errors: {error_count}")
        print(f"  Output: {output_path}")
        print(f"{'='*60}")
        
        # Print statistics
        self.print_statistics(df)
    
    def print_statistics(self, df: pd.DataFrame):
        """Print label distribution statistics."""
        print("\n📊 Label Statistics:")
        print("-" * 60)
        
        for query in self.QUERIES.keys():
            if query in df.columns:
                yes_count = (df[query] == "YES").sum()
                no_count = (df[query] == "NO").sum()
                total = yes_count + no_count
                yes_pct = (yes_count / total * 100) if total > 0 else 0
                
                print(f"{query:25} | YES: {yes_count:4d} ({yes_pct:5.1f}%) | NO: {no_count:4d}")


def estimate_cost(num_reviews: int, avg_review_length: int = 500, model: str = "gpt-4o-mini", batch_mode: bool = False):
    """Estimate API cost for labeling."""
    # OpenAI pricing (as of Dec 2024)
    pricing = {
        "gpt-4o-mini": {
            "input": 0.150 / 1_000_000,   # $0.150 per 1M input tokens
            "output": 0.600 / 1_000_000    # $0.600 per 1M output tokens
        },
        "gpt-4o": {
            "input": 2.50 / 1_000_000,     # $2.50 per 1M input tokens
            "output": 10.00 / 1_000_000    # $10.00 per 1M output tokens
        },
        "gpt-4-turbo": {
            "input": 10.00 / 1_000_000,
            "output": 30.00 / 1_000_000
        }
    }
    
    if model not in pricing:
        print(f"Unknown model: {model}, using gpt-4o-mini pricing")
        model = "gpt-4o-mini"
    
    # Rough estimates
    tokens_per_review = (avg_review_length + 400) / 4  # ~4 chars per token
    input_tokens = num_reviews * tokens_per_review
    output_tokens = num_reviews * 50  # ~50 tokens for JSON response
    
    input_cost = input_tokens * pricing[model]["input"]
    output_cost = output_tokens * pricing[model]["output"]
    total_cost = input_cost + output_cost
    
    # Apply batch discount if enabled
    batch_discount = 0.5 if batch_mode else 1.0
    total_cost *= batch_discount
    
    print(f"\n💰 Estimated Cost ({model}{' - BATCH MODE' if batch_mode else ''}):")
    print(f"  Reviews: {num_reviews:,}")
    print(f"  Input tokens: ~{input_tokens:,.0f}")
    print(f"  Output tokens: ~{output_tokens:,.0f}")
    if batch_mode:
        print(f"  Batch discount: 50% off")
    print(f"  Total cost: ${total_cost:.2f}")
    print(f"  Cost per review: ${total_cost/num_reviews:.5f}\n")
    
    # Show time estimates
    if batch_mode:
        print(f"  ⏱️  Processing time: ~24 hours (batch API)")
    else:
        # Estimate based on rate limits
        seconds = num_reviews * 0.6  # ~600ms per request with processing
        minutes = seconds / 60
        hours = minutes / 60
        if hours > 1:
            print(f"  ⏱️  Processing time: ~{hours:.1f} hours")
        else:
            print(f"  ⏱️  Processing time: ~{minutes:.0f} minutes")
    
    # Show comparison with other modes/models
    if model == "gpt-4o-mini":
        if not batch_mode:
            batch_cost = total_cost * 0.5
            print(f"\n  📊 Batch mode would cost: ${batch_cost:.2f} (50% cheaper, 24h wait)")
        
        gpt4o_cost = (input_tokens * pricing["gpt-4o"]["input"] + 
                      output_tokens * pricing["gpt-4o"]["output"])
        if batch_mode:
            gpt4o_cost *= 0.5
        print(f"  📊 gpt-4o would cost: ${gpt4o_cost:.2f} ({gpt4o_cost/total_cost:.1f}x more expensive)")


def main():
    parser = argparse.ArgumentParser(
        description="Label review relevance using OpenAI API"
    )
    parser.add_argument(
        "--input", 
        required=True,
        help="Input CSV file path"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        choices=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        help="OpenAI model to use (default: gpt-4o-mini, cheapest and fast)"
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="Start from this row (for resuming)"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum rows to process (default: all)"
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=50,
        help="Save checkpoint every N rows"
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Estimate cost and exit"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (1-10, use with caution)"
    )
    
    args = parser.parse_args()
    
    # Cost estimation mode
    if args.estimate_cost:
        df = pd.read_csv(args.input)
        rows_to_process = args.max_rows or len(df)
        estimate_cost(rows_to_process, model=args.model)
        return
    
    # Initialize labeler
    labeler = RelevanceLabeler(model=args.model)
    
    # Process CSV
    labeler.process_csv(
        input_path=args.input,
        output_path=args.output,
        start_row=args.start_row,
        max_rows=args.max_rows,
        checkpoint_interval=args.checkpoint
    )


if __name__ == "__main__":
    main()