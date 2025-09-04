# Terminology Review Agent - Usage Guide

## 🎯 Overview

I've created a comprehensive **Terminology Review Agent** that can validate term candidates from your `New_Terms_Candidates_Clean.json` file using:

- **Web search research** via DuckDuckGo API
- **Autodesk glossary analysis** from your `/glossary/` folder  
- **Industry-specific validation** for CAD/AEC/Manufacturing contexts
- **Comprehensive JSON reporting** for all findings

## 📁 Files Created

### Core Agent Files
- `terminology_review_agent.py` - Main agent with web search capabilities
- `terminology_tool.py` - Existing glossary management (unchanged)

### Processing Scripts
- `validate_new_terms_candidates.py` - Full processing script for large-scale validation
- `quick_validate_sample_terms.py` - Quick test script with sample terms
- `example_terminology_review.py` - Basic usage examples

### Documentation
- `TERMINOLOGY_REVIEW_README.md` - Comprehensive documentation
- `USAGE_GUIDE.md` - This usage guide

## 🚀 Quick Start

### 1. Test with Sample Terms (Recommended First Step)

```bash
# Test the agent with a small sample
python quick_validate_sample_terms.py
```

This validates 5 sample technical terms: `parametric`, `viewport`, `tessellation`, `wireframe`, `boolean`

### 2. Process Your Term Candidates

```bash
# Basic usage - validate first 20 terms
python validate_new_terms_candidates.py --limit 20 --min-frequency 10

# CAD-focused validation with higher frequency threshold
python validate_new_terms_candidates.py --limit 50 --min-frequency 20 --industry CAD

# Full processing (use with caution - will take hours!)
python validate_new_terms_candidates.py --limit 500 --batch-size 10 --industry CAD
```

### 3. Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--limit` | 50 | Maximum number of terms to process |
| `--min-frequency` | 10 | Minimum frequency threshold for terms |
| `--batch-size` | 5 | Terms per batch (smaller = more API calls) |
| `--industry` | General | CAD/AEC/Manufacturing/General |
| `--src-lang` | EN | Source language |
| `--tgt-lang` | CS | Target language |
| `--model` | gpt-4.1 | gpt-5 (slower, better) or gpt-4.1 (faster) |

## 📊 Your Data Analysis

From `New_Terms_Candidates_Clean.json`:
- **Total terms**: 61,371 candidates
- **Total frequency**: 112,329 occurrences
- **Data quality**: High - all verified

### Sample Terms Found
- `autodesk` (frequency: 840)
- `parametric`, `viewport`, `tessellation` (technical terms)
- Many CAD/engineering-related candidates

## 🔍 Validation Process

For each term, the agent:

1. **Glossary Analysis**
   - Checks against existing Autodesk glossaries
   - Finds related terms and translations
   - Identifies industry categories (ACAD, AEC, etc.)

2. **Web Research** 
   - Searches for technical definitions
   - Researches Autodesk-specific usage
   - Validates industry acceptance

3. **Scoring & Recommendations**
   - Calculates validation score (0.0-1.0)
   - Provides status: Recommended/Needs Review/Not Recommended
   - Lists specific reasons and findings

## 📈 Expected Results

### High-Priority Terms (Recommended)
Terms likely to score 0.7+:
- Technical CAD terms: `parametric`, `tessellation`, `wireframe`
- Industry-specific terminology
- Terms found in existing glossaries

### Medium-Priority Terms (Needs Review)
Terms scoring 0.4-0.69:
- General technical terms with some usage
- Terms with partial glossary matches
- Industry terms needing human validation

### Low-Priority Terms (Not Recommended)
Terms scoring below 0.4:
- Common English words
- Brand names without technical meaning
- Terms with limited technical usage

## 🎛️ Recommended Processing Strategy

### Phase 1: Quick Test (5 minutes)
```bash
python quick_validate_sample_terms.py
```

### Phase 2: High-Value Terms (30 minutes)
```bash
python validate_new_terms_candidates.py --limit 30 --min-frequency 50 --industry CAD
```

### Phase 3: Broader Validation (2-3 hours)
```bash
python validate_new_terms_candidates.py --limit 200 --min-frequency 20 --industry CAD --batch-size 8
```

### Phase 4: Full Processing (8+ hours)
```bash
python validate_new_terms_candidates.py --limit 1000 --min-frequency 10 --industry General --batch-size 10
```

## 📄 Output Files

Each run creates:
- `term_validation_batch_N_TIMESTAMP.json` - Individual batch results
- `term_validation_summary_TIMESTAMP.json` - Comprehensive summary
- Individual term reports with web research data

### Sample Output Structure
```json
{
  "validation_summary": {
    "total_terms_processed": 30,
    "recommended_count": 12,
    "needs_review_count": 8,
    "not_recommended_count": 10
  },
  "recommended_terms": [
    {
      "term": "parametric",
      "score": 0.85,
      "key_reasons": ["Found in existing glossaries", "Strong technical usage"]
    }
  ]
}
```

## ⚡ Performance Tips

### Faster Processing
- Use `--model gpt-4.1` instead of `gpt-5`
- Increase `--batch-size` to 8-10
- Set higher `--min-frequency` to focus on common terms

### Better Quality
- Use `--model gpt-5` for best analysis
- Set specific `--industry` context
- Lower `--min-frequency` to catch rare technical terms

### Cost Management
- Start with small `--limit` values
- Use `--min-frequency 20+` to focus on important terms
- Monitor API usage during processing

## 🛠️ Troubleshooting

### Common Issues

**"Azure authentication failed"**
- Check your `.env` file has correct Azure credentials
- Verify Azure AD permissions

**"Web search failed"**
- Check internet connection
- DuckDuckGo API may have rate limits

**"No terms found after filtering"**
- Lower `--min-frequency` threshold
- Check input file format

### Getting Help

1. Start with `quick_validate_sample_terms.py` to test setup
2. Check generated JSON files for detailed error messages
3. Review Azure OpenAI service status if model calls fail

## 📋 Next Steps After Validation

1. **Review Recommended Terms** - Add to official glossaries
2. **Human Review** - Evaluate "Needs Review" terms manually  
3. **Pattern Analysis** - Use insights to improve future term selection
4. **Integration** - Incorporate approved terms into translation workflows

## 🎯 Success Metrics

A successful validation run should show:
- 70%+ of technical terms getting "Recommended" or "Needs Review"
- Detailed web research findings for each term
- Clear rationale for inclusion/exclusion decisions
- Structured data ready for glossary integration

---

**Ready to start?** Run `python quick_validate_sample_terms.py` to test the system!
