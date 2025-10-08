import { useState } from 'react';
import { TerminologyValidationDiagram } from './components/TerminologyValidationDiagram';
import { Card } from './components/ui/card';

export default function App() {
  const [showExplanations, setShowExplanations] = useState(true);

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="mb-2">Terminology Validation and Verification Process</h1>
          <p className="text-muted-foreground">
            Interactive SVG diagram showing the comprehensive process for validating and verifying terminology in English and target languages
          </p>
        </div>

        <div className="grid lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="space-y-4">
            <Card className="p-4">
              <h3 className="mb-4">Process Overview</h3>
              <p className="text-sm text-muted-foreground mb-4">
                This diagram shows the systematic approach for terminology validation including source verification, glossary checks, dictionary validation, frequency analysis, and translation mechanisms.
              </p>
              
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={showExplanations}
                  onChange={(e) => setShowExplanations(e.target.checked)}
                  className="rounded"
                />
                <span>Show Explanations</span>
              </label>
            </Card>

            <Card className="p-4">
              <h4 className="mb-2">Key Features</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Click elements for details</li>
                <li>• Source verification</li>
                <li>• Glossary validation</li>
                <li>• Dictionary checks</li>
                <li>• Frequency analysis</li>
                <li>• Multi-language translation</li>
                <li>• Review workflow</li>
              </ul>
            </Card>

            <Card className="p-4">
              <h4 className="mb-2">Translation Models</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• NLLB (No Language Left Behind)</li>
                <li>• AYA 101</li>
                <li>• Generic translation</li>
                <li>• Up to 200 languages</li>
              </ul>
            </Card>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            <TerminologyValidationDiagram showExplanations={showExplanations} />
          </div>
        </div>
      </div>
    </div>
  );
}