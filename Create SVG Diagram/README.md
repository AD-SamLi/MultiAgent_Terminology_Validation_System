# 🎨 Multi-Agent Terminology Validation System - Interactive Diagram

An interactive visualization tool for understanding the **9-step terminology validation pipeline** of the Multi-Agent Terminology Validation System.

## 📊 Overview

This interactive diagram visualizes the complete workflow of how the system processes 10,997+ terms through validation, translation, and quality assessment.

### System Architecture Visualization

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    INPUT: Term_Extracted_result.csv                      │
│                           (10,997 terms)                                  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Data Collection                                                 │
│  ├─ Load and combine terminology data                                   │
│  ├─ Clean and verify terms                                              │
│  └─ Output: Combined_Terms_Data.csv, Cleaned_Terms_Data.csv             │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Glossary Analysis (Parallel Processing)                        │
│  ├─ Check against existing glossaries                                   │
│  ├─ Use 16 CPU cores for parallel batch processing                      │
│  ├─ ~687 terms/batch, ~43 terms/worker                                  │
│  ├─ AI-powered Terminology Agent analysis                               │
│  ├─ Classify: EXISTING (9,769) vs NEW (1,228)                           │
│  └─ Output: Glossary_Analysis_Results.json                              │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3: New Term Identification                                         │
│  ├─ Process only NEW terms (1,228)                                       │
│  ├─ Dictionary validation (NLTK WordNet)                                │
│  ├─ FastDictionaryAgent analysis                                        │
│  └─ Output: New_Terms_Candidates_With_Dictionary.json                   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Frequency Analysis                                              │
│  ├─ Filter high-frequency terms (≥2 occurrences)                        │
│  ├─ Statistical filtering: 1,228 → 429 dictionary terms                 │
│  └─ Output: High_Frequency_Terms.json (429 terms)                       │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 5: Translation Process (GPU-Accelerated)                          │
│  ├─ Multi-language translation (200+ languages)                         │
│  ├─ NLLB-200-1.3B model with multi-GPU support                          │
│  ├─ Dynamic resource allocation (up to 3 GPUs)                          │
│  ├─ Ultra-optimized smart runner                                        │
│  ├─ Checkpoint-based resumption                                         │
│  └─ Output: Translation_Results.json (429 terms × 200 languages)        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 6: Verification                                                    │
│  ├─ Quality assessment for translations                                 │
│  ├─ Language consistency verification                                   │
│  ├─ Translatability score calculation                                   │
│  └─ Output: Verified_Translation_Results.json                           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 7: Final Review & Decision (AI Agents)                            │
│  ├─ Modern batch processing (1,087+ batch files)                        │
│  ├─ AI agent validation (smolagents framework)                          │
│  ├─ ML-based quality scoring                                            │
│  ├─ Translatability analysis                                            │
│  ├─ Decision categories:                                                │
│  │   • APPROVED: 31.6%                                                  │
│  │   • CONDITIONALLY_APPROVED: 54.7%                                    │
│  │   • NEEDS_REVIEW: 12.9%                                              │
│  │   • REJECTED: 0.7%                                                   │
│  └─ Output: Final_Terminology_Decisions.json                            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 8: Audit Record                                                   │
│  ├─ Complete audit trail generation                                     │
│  ├─ Process statistics and metadata                                     │
│  └─ Output: Complete_Audit_Record.json                                  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 9: CSV Export (Azure OpenAI GPT-4.1)                              │
│  ├─ Export approved terms (7,503 terms)                                 │
│  ├─ Professional context generation using GPT-4.1                       │
│  ├─ Parallel processing (20 workers)                                    │
│  ├─ CSV format: source, target, description, context                    │
│  └─ Output: Approved_Terms_Export.csv                                   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │  FINAL OUTPUT       │
                      │  7,503 approved     │
                      │  professional CSV   │
                      └─────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ or npm/pnpm/bun
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

```bash
cd "Create SVG Diagram"

# Install dependencies
npm install

# Start development server
npm run dev
```

### Build for Production

```bash
npm run build
npm run preview
```

## 📁 Project Structure

```
Create SVG Diagram/
├── src/
│   ├── components/
│   │   ├── TerminologyValidationDiagram.tsx  # Main diagram component
│   │   ├── DiagramViewer.tsx                 # Interactive viewer
│   │   └── ui/                               # UI components (shadcn/ui)
│   ├── guidelines/
│   │   └── Guidelines.md                     # Design guidelines
│   ├── App.tsx                               # Main application
│   └── main.tsx                              # Entry point
├── index.html
├── package.json
├── vite.config.ts
└── README.md
```

## 🎯 Features

- **Interactive Visualization**: Click to explore each processing step
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Updates**: Live updates as the system processes
- **Step Details**: Detailed information for each pipeline step
- **Performance Metrics**: Visual representation of system performance
- **Export Options**: Export diagram as SVG or PNG

## 🛠️ Tech Stack

- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **UI Components**: shadcn/ui (Radix UI primitives)
- **Styling**: Tailwind CSS
- **Diagrams**: Custom SVG components

## 📊 System Metrics Visualized

| Metric | Value | Visualization |
|--------|-------|---------------|
| **Total Terms** | 10,997 | Input flow |
| **Existing Terms** | 9,769 | Step 2 classification |
| **New Terms** | 1,228 | Step 3 processing |
| **High Frequency** | 429 | Step 4 filtering |
| **Approved Terms** | 7,503 | Step 9 export |
| **Approval Rate** | 86.4% | Quality metrics |
| **Processing Speed** | ~11 terms/sec | Step 2 parallel |
| **Translation Coverage** | 200+ languages | Step 5 output |

## 🔧 Development

### Adding New Visualizations

1. Create component in `src/components/`
2. Import into `TerminologyValidationDiagram.tsx`
3. Add interactive features in `DiagramViewer.tsx`

### Styling Guidelines

- Follow Tailwind CSS conventions
- Use shadcn/ui components for consistency
- Maintain responsive design principles
- See `src/guidelines/Guidelines.md` for details

## 📖 Documentation

For comprehensive system documentation, see:
- **README.md** (root): System overview and usage
- **TECHNICAL_DOCUMENTATION.md**: Detailed technical specifications
- **CHANGELOG.md**: Version history and updates

## 🤝 Contributing

Improvements to the visualization are welcome:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see root LICENSE file for details

## 🆘 Support

For issues or questions:
- Check the main system documentation
- Review `Guidelines.md` for design standards
- Create GitHub issues for bugs

---

**Version**: 1.0.0  
**Last Updated**: October 3, 2025  
**System**: Multi-Agent Terminology Validation System  
**Purpose**: Interactive visualization of 9-step pipeline
