# LLM-Powered Relevance Labeler

Automated tool to classify review relevance using OpenAI's GPT models. Creates evaluation gold sets for RAG systems by labeling reviews against multiple queries simultaneously.

## 🎯 What It Does

Takes a CSV of reviews and automatically labels each review as relevant (YES/NO) for multiple search queries using GPT-4o-mini or GPT-4o.

**Example:**
- **Query**: "dog got sick"
- **Review**: "My dog vomited after eating this food"
- **Label**: YES

Perfect for building evaluation datasets for RAG/semantic search systems.

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- CSV file with columns: `Id`, `Summary`, `Text`

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install openai pandas python-dotenv tqdm
```

### 2. Setup API Key

Create a `.env` file in the same directory:

```env
OPENAI_API_KEY=sk-your-api-key-here
```

Get your API key from: https://platform.openai.com/api-keys

### 3. Prepare Your CSV

Your CSV should have these columns:
- **Id**: Unique identifier for each review
- **Summary**: Review title/summary (can be empty)
- **Text**: Full review text

Example:
```csv
Id,Summary,Text
1,Great Dog Food,I have bought several of this dog food...
2,My dog got sick,Awful product. My dog vomited after...
```

### 4. Run the Labeler

```bash
# Estimate cost first (always recommended)
python label_relevance.py --input reviews.csv --output labeled.csv --estimate-cost

# Label all reviews
python label_relevance.py --input reviews.csv --output labeled.csv
```

## 💰 Cost & Time Estimates

| Reviews | Model        | Cost      | Time      |
|---------|--------------|-----------|-----------|
| 100     | gpt-4o-mini  | ~$0.02    | ~1 min    |
| 1,000   | gpt-4o-mini  | ~$0.20    | ~10 min   |
| 2,187   | gpt-4o-mini  | ~$0.50    | ~25 min   |
| 10,000  | gpt-4o-mini  | ~$2.00    | ~2 hours  |
| 50,000  | gpt-4o-mini  | ~$10.00   | ~8 hours  |

*gpt-4o costs ~17x more but provides higher accuracy*

## 📖 Usage Examples

### Basic Usage

```bash
# Label all reviews with gpt-4o-mini (default)
python label_relevance.py --input reviews.csv --output labeled.csv
```

### Test First (Recommended)

```bash
# Label first 50 reviews to test quality
python label_relevance.py --input reviews.csv --output test.csv --max-rows 50

# Check quality manually, then run full dataset
python label_relevance.py --input reviews.csv --output labeled.csv
```

### Use Higher Quality Model

```bash
# Use gpt-4o for better accuracy
python label_relevance.py --input reviews.csv --output labeled.csv --model gpt-4o
```

### Process in Batches

```bash
# Process 1000 at a time
python label_relevance.py --input reviews.csv --output labeled.csv --start-row 0 --max-rows 1000
python label_relevance.py --input reviews.csv --output labeled.csv --start-row 1000 --max-rows 1000
python label_relevance.py --input reviews.csv --output labeled.csv --start-row 2000 --max-rows 1000
```

### Resume After Interruption

```bash
# If interrupted at row 850, resume from there
python label_relevance.py --input reviews.csv --output labeled.csv --start-row 850
```

## 🔧 Configuration

### Queries

Edit the `QUERIES` dictionary in `label_relevance.py` to customize your queries:

```python
QUERIES = {
    "Good Dog Food": "Reviews mentioning positive experiences...",
    "Bad dog food": "Reviews mentioning negative experiences...",
    "dog got sick": "Reviews mentioning health issues...",
    # Add your own queries here
}
```

### Checkpointing

The script auto-saves progress every 50 rows by default. Change this:

```bash
python label_relevance.py --input reviews.csv --output labeled.csv --checkpoint 100
```

### Text Truncation

Reviews are truncated to 1,000 characters to save costs. To change this, edit line in the code:

```python
**Review Text:** {text[:1000]}...  # Change 1000 to your desired length
```

## 📊 Output Format

The output CSV adds YES/NO columns for each query:

```csv
Id,Summary,Text,Good Dog Food,Bad dog food,dog got sick,...
1,Great Food,I love this...,YES,NO,NO,...
2,Dog vomited,Awful product...,NO,YES,YES,...
```

## ✅ Quality Validation

After labeling, spot-check results:

```python
import pandas as pd

df = pd.read_csv("labeled.csv")

# Check "dog got sick" labels
sick_reviews = df[df["dog got sick"] == "YES"].sample(5)
print(sick_reviews[["Id", "Summary", "Text"]])

# Check label distribution
for query in ["Good Dog Food", "Bad dog food", "dog got sick"]:
    yes_count = (df[query] == "YES").sum()
    print(f"{query}: {yes_count} YES labels")
```

## 🛠️ Command Line Options

```bash
python label_relevance.py [OPTIONS]

Required:
  --input PATH          Input CSV file
  --output PATH         Output CSV file

Optional:
  --model MODEL         OpenAI model: gpt-4o-mini (default), gpt-4o, gpt-4-turbo
  --start-row N         Start from row N (for resuming)
  --max-rows N          Process only N rows
  --checkpoint N        Save every N rows (default: 50)
  --estimate-cost       Estimate cost and exit (no processing)

Examples:
  python label_relevance.py --input reviews.csv --output labeled.csv
  python label_relevance.py --input reviews.csv --output labeled.csv --model gpt-4o
  python label_relevance.py --input reviews.csv --output labeled.csv --max-rows 100
  python label_relevance.py --input reviews.csv --output labeled.csv --estimate-cost
```

## 🐛 Troubleshooting

### "OPENAI_API_KEY not found"
- Make sure `.env` file exists in same directory
- Check `.env` has: `OPENAI_API_KEY=sk-...`
- No quotes needed around the key

### "Missing required columns"
- Your CSV must have: `Id`, `Summary`, `Text`
- Column names are case-sensitive

### API Rate Limits
- gpt-4o-mini: 10,000 requests/min (very high)
- If you hit limits, the script will retry automatically
- Or add delays: edit `time.sleep(0.1)` to `time.sleep(1)`

### Out of Memory
- Process in smaller batches with `--max-rows`
- Example: `--max-rows 1000`

### Poor Label Quality
- Try gpt-4o instead: `--model gpt-4o`
- Adjust query descriptions in the code
- Increase text truncation: `text[:1500]` instead of `text[:1000]`

## 📈 Best Practices

1. **Always estimate cost first** with `--estimate-cost`
2. **Test on 50-100 reviews** before running full dataset
3. **Spot check 10-20 labels** manually to validate quality
4. **Use checkpointing** for large datasets (it's automatic)
5. **Start with gpt-4o-mini** (cheap), upgrade to gpt-4o if needed

## 💡 Tips for Large Datasets

### For 10k+ Reviews

```bash
# Run overnight with logging
nohup python label_relevance.py --input reviews.csv --output labeled.csv > labeling.log 2>&1 &

# Check progress
tail -f labeling.log
```

### For 50k+ Reviews

```bash
# Process in 10k chunks
for i in {0..40000..10000}; do
  python label_relevance.py --input reviews.csv --output labeled.csv --start-row $i --max-rows 10000
done
```

## 📝 Statistics Output

At the end of processing, you'll see:

```
📊 Label Statistics:
------------------------------------------------------------
Good Dog Food            | YES:  450 ( 20.6%) | NO: 1737
Bad dog food             | YES:  312 ( 14.3%) | NO: 1875
dog got sick             | YES:   89 (  4.1%) | NO: 2098
dog is picky eater       | YES:  156 (  7.1%) | NO: 2031
delivery issue           | YES:   43 (  2.0%) | NO: 2144
grain free dog food      | YES:  234 ( 10.7%) | NO: 1953
refund                   | YES:   21 (  1.0%) | NO: 2166
```

## 🔗 Next Steps

After labeling, use this data to:

1. **Evaluate your RAG system**:
   - Calculate Recall@K, Precision@K
   - Compare different embedding models
   - Tune retrieval parameters

2. **Build test sets**:
   - Split into train/validation/test
   - Create difficulty tiers
   - Stratify by query type

3. **Improve retrieval**:
   - Identify problematic queries
   - Add query expansion
   - Implement reranking

## 📄 License

MIT

## 🤝 Contributing

Issues and PRs welcome! This is a utility tool for RAG evaluation.

## ⚠️ Important Notes

- **API costs**: Always use `--estimate-cost` first
- **Quality**: Spot-check results before using in production
- **Checkpoints**: The script auto-saves, so interrupting is safe
- **Rate limits**: Script has built-in delays and retries
- **Privacy**: Review texts are sent to OpenAI API (check their privacy policy)

## 📧 Support

For issues or questions, please refer to OpenAI's documentation:
- API Docs: https://platform.openai.com/docs
- Pricing: https://openai.com/pricing
- Rate Limits: https://platform.openai.com/docs/guides/rate-limits