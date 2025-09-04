# Terminology Review Agent

An advanced AI-powered agent that validates term candidates for Autodesk by combining internal glossary analysis with web search research to determine industry usage and technical validity.

## 🌟 Features

### Core Capabilities
- **Term Validation**: Comprehensive validation of terminology candidates using multiple data sources
- **Web Search Integration**: Real-time industry research via DuckDuckGo search API
- **Autodesk Context Analysis**: Leverages existing Autodesk glossaries to understand terminology patterns
- **Industry-Specific Research**: Focused analysis for CAD, AEC, and Manufacturing domains
- **Batch Processing**: Efficient validation of multiple terms simultaneously
- **JSON Output**: Structured data export for integration with other systems

### Validation Process
1. **Glossary Analysis**: Checks term against existing Autodesk terminology databases
2. **Industry Research**: Web searches for technical definitions and usage patterns
3. **Context Evaluation**: Analyzes term relevance to Autodesk product ecosystems
4. **Scoring System**: Calculates validation scores based on multiple criteria
5. **Recommendations**: Provides actionable insights for term inclusion decisions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Azure OpenAI access (GPT-5 or GPT-4.1)
- Internet connection for web search functionality

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

### Basic Usage

```python
from terminology_review_agent import TerminologyReviewAgent

# Initialize the agent
agent = TerminologyReviewAgent(
    glossary_folder="./glossary",
    model_name="gpt-5"  # or "gpt-4.1" for faster processing
)

# Validate a single term
result = agent.validate_term_candidate(
    term="parametric modeling",
    src_lang="EN",
    tgt_lang="DE", 
    industry_context="CAD",
    save_to_file="validation_result.json"
)

# Batch validate multiple terms
terms = ["mesh topology", "boolean operations", "NURBS surface"]
batch_result = agent.batch_validate_terms(
    terms=terms,
    industry_context="CAD",
    save_to_file="batch_validation.json"
)
```

## 📊 Validation Scoring

The agent uses a comprehensive scoring system:

| Score Range | Status | Meaning |
|-------------|--------|---------|
| 0.7 - 1.0 | ✅ Recommended | Strong evidence of technical validity and industry usage |
| 0.4 - 0.69 | ⚠️ Needs Review | Some evidence found, requires human evaluation |
| 0.0 - 0.39 | ❌ Not Recommended | Limited evidence of technical usage in target domain |

### Scoring Factors
- **Existing Translations** (40%): Term found in current Autodesk glossaries
- **Related Terms** (20%): Similar terms exist in glossaries
- **Web Presence** (20%): Strong technical definition found online
- **Autodesk Context** (20%): Term used in Autodesk product contexts

## 🔧 Available Actions

### 1. validate_term
Comprehensive validation of a single term candidate.

```python
agent.validate_term_candidate(
    term="surface continuity",
    src_lang="EN",
    tgt_lang="FR",
    industry_context="CAD"
)
```

### 2. batch_validate_terms  
Process multiple terms efficiently.

```python
agent.batch_validate_terms(
    terms=["spline interpolation", "mesh generation", "parametric constraints"],
    industry_context="CAD"
)
```

### 3. research_industry_usage
Deep research on industry usage patterns.

```python
agent.research_industry_usage(
    term="boolean operations",
    industry_context="CAD"
)
```

### 4. analyze_glossary_patterns
Analyze existing glossary characteristics.

```python
agent.analyze_glossary_patterns()
```

### 5. generate_comprehensive_report
Full validation report with extended research.

```python
agent.generate_comprehensive_report(
    term="NURBS surface",
    industry_context="CAD"
)
```

## 📁 Output Format

All results are saved as structured JSON files:

```json
{
  "metadata": {
    "generated_by": "TerminologyReviewAgent",
    "timestamp": "2024-01-15T10:30:00Z",
    "total_results": 1
  },
  "results": [{
    "term": "parametric modeling",
    "validation_score": 0.85,
    "status": "recommended",
    "autodesk_analysis": {
      "existing_translations": {"EN-DE": "parametrische Modellierung"},
      "related_terms": [...],
      "industry_categories": ["CAD"]
    },
    "web_research": {
      "general_industry": "...",
      "autodesk_specific": "...",
      "industry_context": "..."
    },
    "recommendations": [
      "Term found in existing Autodesk glossaries",
      "Strong web presence in technical contexts"
    ]
  }]
}
```

## 🏭 Industry Contexts

### CAD (Computer-Aided Design)
- AutoCAD, Inventor, Fusion 360 terminology
- Technical drawing and modeling terms
- Geometric and parametric concepts

### AEC (Architecture, Engineering, Construction) 
- Revit, Civil 3D terminology
- Building and infrastructure terms
- BIM and construction concepts

### Manufacturing
- Manufacturing terminology
- Production and assembly terms
- Industrial process concepts

### General
- Cross-industry technical terms
- Common software terminology
- Universal engineering concepts

## 🌐 Web Search Integration

The agent uses DuckDuckGo search to research:
- Technical definitions and usage
- Industry-specific applications  
- Autodesk product documentation references
- Related terminology and concepts

### Search Strategies
- **General Industry**: `"{term}" CAD software engineering technical terminology`
- **Autodesk Specific**: `"{term}" Autodesk AutoCAD Revit Inventor Maya`
- **Industry Context**: `"{term}" {industry} industry terminology definition`

## 🔍 Glossary Integration

The agent analyzes existing Autodesk glossaries from:
- `glossary/data/general/english-to-others/` - English to target language translations
- `glossary/data/general/others-to-english/` - Target language to English translations  
- `glossary/data/ui-element/` - User interface terminology
- Product-specific glossaries (ACAD, AEC, CORE, DNM, MNE)

## ⚙️ Configuration

### Environment Variables
```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_TENANT_ID=your-tenant-id
```

### Model Selection
- **GPT-5**: Best results, slower processing (30-60+ seconds per task)
- **GPT-4.1**: Good results, faster processing (10-20 seconds per task)

## 📈 Use Cases

### Translation Project Management
- Validate new terminology before adding to glossaries
- Ensure consistency across product lines
- Research industry-standard translations

### Quality Assurance
- Verify technical accuracy of proposed terms
- Check alignment with Autodesk product terminology
- Identify potential translation issues

### Terminology Development
- Discover gaps in existing glossaries
- Research emerging technical concepts
- Standardize terminology across teams

## 🤝 Integration

The agent can be integrated into existing workflows:

### CI/CD Pipeline
```python
# Validate terminology in automated builds
def validate_new_terms(term_list):
    agent = TerminologyReviewAgent("./glossary")
    results = agent.batch_validate_terms(term_list)
    return parse_validation_results(results)
```

### Translation Management Systems
```python
# API endpoint for term validation
@app.route('/validate-term', methods=['POST'])
def validate_term():
    term = request.json['term']
    result = agent.validate_term_candidate(term)
    return jsonify(result)
```

## 🛠️ Troubleshooting

### Common Issues

**Azure Authentication Errors**
- Verify environment variables are set correctly
- Check Azure AD permissions for Cognitive Services
- Ensure token hasn't expired

**Web Search Failures**
- Check internet connectivity
- Verify DuckDuckGo service availability
- Review rate limiting settings

**Glossary Loading Issues**
- Confirm glossary folder structure matches expected format
- Check CSV file encoding (should be UTF-8)
- Verify file permissions

### Performance Optimization

**Faster Processing**
- Use GPT-4.1 instead of GPT-5 for development
- Reduce `max_results` in DuckDuckGo search tool
- Implement caching for repeated terms

**Better Results**
- Use specific industry contexts
- Provide comprehensive glossary data
- Fine-tune search queries for domain-specific results

## 📝 Example Output

Running the example script generates several JSON files with detailed validation data:

- `validation_result.json` - Single term validation
- `batch_validation.json` - Multiple term results  
- `industry_research.json` - Deep industry usage research
- `glossary_pattern_analysis.json` - Glossary characteristics analysis
- `comprehensive_report.json` - Full validation report

## 🔮 Future Enhancements

- **Multi-language Support**: Expand beyond current language pairs
- **Custom Industry Contexts**: Add domain-specific validation rules
- **Machine Learning Integration**: Train models on validation patterns
- **Real-time Collaboration**: Multi-user terminology review workflows
- **Advanced Analytics**: Terminology trend analysis and reporting

---

For questions, issues, or contributions, please refer to the project documentation or contact the development team.
