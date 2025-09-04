# Terminology Agent with smolagents and Azure OpenAI

A sophisticated terminology management agent that combines:
- **smolagents**: Multi-agent framework with tool integration
- **Azure OpenAI GPT-5**: Advanced reasoning capabilities  
- **Custom Terminology Tool**: Multilingual glossary management

## 🚀 Features

- **Independent Agent**: Self-contained with no external dependencies
- **Multi-language Support**: 100+ language pairs from Autodesk glossaries
- **Intelligent Analysis**: Find terminology terms in text automatically
- **Consistent Translation**: Apply glossary-based replacements
- **DNT Support**: Handle "Do Not Translate" terms correctly
- **Azure Integration**: Uses Azure AD authentication with GPT-5 or GPT-4.1

## 📁 File Structure

```
├── terminology_agent.py       # Main independent agent (self-contained)
├── run_terminology_agent.py   # Interactive runner with menu
├── test_terminology_setup.py  # Setup verification
├── terminology_tool.py        # Core terminology logic
└── glossary/                  # Your glossary CSV files
    └── data/
        ├── general/english-to-others/
        ├── general/others-to-english/  
        └── ui-element/
```

## 🔧 Setup

1. **Install Dependencies**
   ```bash
   pip install smolagents[toolkit] azure-identity python-dotenv pandas
   ```

2. **Set Environment Variables** (create `.env` file)
   ```
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_CLIENT_ID=your-client-id
   AZURE_CLIENT_SECRET=your-client-secret
   AZURE_TENANT_ID=your-tenant-id
   ```

3. **Verify Setup**
   ```bash
   python test_terminology_setup.py
   ```

## 🎯 Usage

### Interactive Mode
```bash
python run_terminology_agent.py
```

### Programmatic Usage
```python
from terminology_agent import TerminologyAgent

# Initialize agent (choose gpt-5 or gpt-4.1)
agent = TerminologyAgent("glossary", model_name="gpt-5")

# Get glossary overview
overview = agent.get_glossary_overview()

# Analyze text for terminology
text = "The AutoCAD file contains vertex data"
analysis = agent.analyze_text_terminology(text, "EN", "CS")

# Translate with terminology consistency
translation = agent.translate_with_terminology(text, "EN", "CS")
```

## 🛠️ Core Components

### 1. TerminologySmolTool
Wraps the terminology tool as a smolagents Tool with actions:
- `get_languages`: List available language codes
- `get_language_pairs`: Show source-target combinations  
- `find_used_terms`: Find glossary terms in text
- `replace_terms`: Apply terminology replacements
- `get_glossary_info`: Complete glossary overview

### 2. PatchedAzureOpenAIServerModel
GPT-5 compatible model that:
- Removes unsupported `stop` parameter
- Handles temperature restrictions
- Converts `max_tokens` to `max_completion_tokens`

### 3. TerminologyAgent
Main agent class with methods:
- `get_glossary_overview()`: Comprehensive glossary analysis
- `analyze_text_terminology()`: Find terms in text
- `translate_with_terminology()`: Apply consistent translations
- `run_terminology_task()`: Execute custom queries

## 🌟 Key Advantages

1. **Independence**: No dependency on test files or external modules
2. **Robustness**: Built-in retry logic for Azure OpenAI issues
3. **Flexibility**: Works with both GPT-5 and GPT-4.1
4. **Intelligence**: Uses AI to understand context and apply terminology
5. **Scalability**: Supports hundreds of language pairs and thousands of terms

## 📊 Example Outputs

The agent can provide detailed analysis like:
- "Found 3 terminology terms: 'vertex' → 'vrchol', 'compile' → 'kompilovat'"
- "Applied 5 terminology replacements with 100% consistency"
- "Detected 2 DNT terms that should remain unchanged"

## 🔄 Integration with smolagents

Uses [smolagents framework](https://huggingface.co/docs/smolagents/main/en/index) features:
- **CodeAgent**: Executes code-based reasoning
- **Custom Tools**: Integrates terminology management
- **Multi-step Operations**: Complex terminology workflows
- **Azure OpenAI**: Enterprise-grade model hosting

Perfect for professional translation workflows requiring terminology consistency!

