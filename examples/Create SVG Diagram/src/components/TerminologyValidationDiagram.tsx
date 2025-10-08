import { useState } from 'react';
import { Card } from './ui/card';
import { Badge } from './ui/badge';

interface TerminologyValidationDiagramProps {
  showExplanations: boolean;
}

interface ProcessStep {
  id: string;
  title: string;
  description: string;
  type: 'start' | 'process' | 'decision' | 'data' | 'agent' | 'translation' | 'end' | 'storage';
  details: string;
}

const processSteps: ProcessStep[] = [
  {
    id: 'termList',
    title: 'Term List',
    description: 'Collection of terms that need validation',
    type: 'start',
    details: 'The process begins with a collection of terms that need validation. This is the initial input containing terminology to be processed.'
  },
  {
    id: 'sourceCheck',
    title: 'Check Source',
    description: 'Verify terminology exists in original sources',
    type: 'decision',
    details: 'Verify if the terminology exists in the original source materials to confirm its authenticity and relevance.'
  },
  {
    id: 'glossaryCheck',
    title: 'Terminology Glossary',
    description: 'Compare against existing terminology glossary',
    type: 'data',
    details: 'Compare the new terms against the existing terminology glossary to check for duplicates or conflicts.'
  },
  {
    id: 'glossaryAgent',
    title: 'Glossary Agent',
    description: 'Specialized agent managing terminology glossary',
    type: 'agent',
    details: 'A specialized agent that manages the terminology glossary and assists in the validation process.'
  },
  {
    id: 'mtGlossary',
    title: 'MT Glossary',
    description: 'Machine Translation glossary reference',
    type: 'data',
    details: 'Machine Translation glossary that contains approved terminology for reference during the validation process.'
  },
  {
    id: 'newTerminology',
    title: 'New Terminology',
    description: 'Terms that passed initial verification',
    type: 'process',
    details: 'Terms that have passed initial verification are marked as new terminology for further processing.'
  },
  {
    id: 'dictionaryCheck',
    title: 'Dictionary Check',
    description: 'Verify against most up-to-date dictionary',
    type: 'decision',
    details: 'Verify if the new terminology exists in the most current dictionary resources for both English and target languages.'
  },
  {
    id: 'languageOptional',
    title: 'English & Target Language',
    description: 'Language verification (optional)',
    type: 'process',
    details: 'The verification can be performed in both English and the target language as needed.'
  },
  {
    id: 'frequencyFilter',
    title: 'Frequency Filter',
    description: 'Filter terms with frequency > 2',
    type: 'decision',
    details: 'Only terms that appear with a frequency greater than 2 are considered for immediate inclusion.'
  },
  {
    id: 'storage',
    title: 'Store Low Frequency',
    description: 'Store terms with frequency = 1 for future',
    type: 'storage',
    details: 'Terms that appear only once are stored for future reference and will be reconsidered if they appear again.'
  },
  {
    id: 'genericTranslation',
    title: 'Generic Translation',
    description: 'Use generic translation approach',
    type: 'translation',
    details: 'If a term passes all checks but doesn\'t have an established translation, a generic translation approach is used.'
  },
  {
    id: 'language1',
    title: 'Translate Language 1',
    description: 'Begin translation for first language',
    type: 'translation',
    details: 'Begin translation process for the first target language.'
  },
  {
    id: 'language200',
    title: 'Translate Language 200',
    description: 'Continue for multiple languages',
    type: 'translation',
    details: 'Continue translation process for multiple languages (up to 200 different languages).'
  },
  {
    id: 'translationModels',
    title: 'NLLB & AYA 101',
    description: 'Translation models for accuracy',
    type: 'agent',
    details: 'Utilize No Language Left Behind (NLLB) and AYA 101 translation models for accurate translations.'
  },
  {
    id: 'languageVerify',
    title: 'Language Verification',
    description: 'Verify source and target language matching',
    type: 'decision',
    details: 'Ensure that the source and target languages are correctly identified and matched.'
  },
  {
    id: 'timestamp',
    title: 'Timestamp + Data',
    description: 'Record timestamp and term data',
    type: 'process',
    details: 'Record timestamp information along with term data for tracking and auditing purposes.'
  },
  {
    id: 'reviewAgent',
    title: 'Web Review Agent',
    description: 'Final terminology review',
    type: 'agent',
    details: 'A specialized agent that performs the final review of the terminology before approval.'
  },
  {
    id: 'decision',
    title: 'Yes/No Decision',
    description: 'Final approval or rejection',
    type: 'decision',
    details: 'Final determination on whether to approve or reject the terminology based on all validation checks.'
  },
  {
    id: 'failed',
    title: 'Failed',
    description: 'Rejected terminology',
    type: 'end',
    details: 'Terms that do not pass the validation process are marked as failed and excluded from the glossary.'
  },
  {
    id: 'approved',
    title: 'Approved',
    description: 'Successfully validated terminology',
    type: 'end',
    details: 'Terms that pass all validation checks are approved and added to the terminology system.'
  }
];

export function TerminologyValidationDiagram({ showExplanations }: TerminologyValidationDiagramProps) {
  const [selectedStep, setSelectedStep] = useState<string | null>(null);

  const getStepColor = (type: string) => {
    switch (type) {
      case 'start': return { fill: '#4ade80', stroke: '#16a34a' };
      case 'process': return { fill: '#60a5fa', stroke: '#1d4ed8' };
      case 'decision': return { fill: '#fbbf24', stroke: '#d97706' };
      case 'data': return { fill: '#f87171', stroke: '#dc2626' };
      case 'agent': return { fill: '#a78bfa', stroke: '#7c3aed' };
      case 'translation': return { fill: '#fb7185', stroke: '#e11d48' };
      case 'storage': return { fill: '#34d399', stroke: '#059669' };
      case 'end': return { fill: '#f87171', stroke: '#dc2626' };
      default: return { fill: '#9ca3af', stroke: '#6b7280' };
    }
  };

  const renderTerminologyValidationDiagram = () => (
    <svg viewBox="0 0 1400 1280" className="w-full h-full">
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 10 3, 0 6" fill="#374151" />
        </marker>
        <marker id="arrowRed" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 10 3, 0 6" fill="#dc2626" />
        </marker>
        <marker id="arrowGreen" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 10 3, 0 6" fill="#16a34a" />
        </marker>
      </defs>

      {/* Term List - Start */}
      <rect
        x="50" y="50" width="140" height="60" rx="8"
        fill={getStepColor('start').fill} stroke={getStepColor('start').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('termList')}
      />
      <text x="120" y="85" textAnchor="middle" className="text-sm fill-black">Term List</text>

      {/* Check Source */}
      <polygon
        points="320,60 380,90 320,120 260,90"
        fill={getStepColor('decision').fill} stroke={getStepColor('decision').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('sourceCheck')}
      />
      <text x="320" y="95" textAnchor="middle" className="text-xs fill-black">Check Source</text>

      {/* Terminology Glossary */}
      <rect
        x="500" y="50" width="140" height="60" rx="8"
        fill={getStepColor('data').fill} stroke={getStepColor('data').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('glossaryCheck')}
      />
      <text x="570" y="85" textAnchor="middle" className="text-xs fill-white">Terminology Glossary</text>

      {/* Glossary Agent */}
      <rect
        x="700" y="50" width="140" height="60" rx="8"
        fill={getStepColor('agent').fill} stroke={getStepColor('agent').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('glossaryAgent')}
      />
      <text x="770" y="85" textAnchor="middle" className="text-xs fill-white">Glossary Agent</text>

      {/* MT Glossary */}
      <rect
        x="900" y="50" width="140" height="60" rx="8"
        fill={getStepColor('data').fill} stroke={getStepColor('data').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('mtGlossary')}
      />
      <text x="970" y="85" textAnchor="middle" className="text-xs fill-white">MT Glossary</text>

      {/* New Terminology */}
      <rect
        x="250" y="170" width="140" height="60" rx="8"
        fill={getStepColor('process').fill} stroke={getStepColor('process').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('newTerminology')}
      />
      <text x="320" y="205" textAnchor="middle" className="text-xs fill-white">New Terminology</text>

      {/* Dictionary Check */}
      <polygon
        points="320,280 390,310 320,340 250,310"
        fill={getStepColor('decision').fill} stroke={getStepColor('decision').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('dictionaryCheck')}
      />
      <text x="320" y="315" textAnchor="middle" className="text-xs fill-black">Dictionary Check</text>

      {/* English & Target Language */}
      <rect
        x="500" y="280" width="140" height="60" rx="8"
        fill={getStepColor('process').fill} stroke={getStepColor('process').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('languageOptional')}
      />
      <text x="570" y="315" textAnchor="middle" className="text-xs fill-white">English & Target</text>

      {/* Frequency Filter */}
      <polygon
        points="320,400 390,430 320,460 250,430"
        fill={getStepColor('decision').fill} stroke={getStepColor('decision').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('frequencyFilter')}
      />
      <text x="320" y="435" textAnchor="middle" className="text-xs fill-black">Frequency &gt; 2?</text>

      {/* Store Low Frequency */}
      <rect
        x="50" y="400" width="140" height="60" rx="8"
        fill={getStepColor('storage').fill} stroke={getStepColor('storage').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('storage')}
      />
      <text x="120" y="435" textAnchor="middle" className="text-xs fill-black">Store Freq=1</text>

      {/* Generic Translation */}
      <rect
        x="250" y="520" width="140" height="60" rx="8"
        fill={getStepColor('translation').fill} stroke={getStepColor('translation').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('genericTranslation')}
      />
      <text x="320" y="555" textAnchor="middle" className="text-xs fill-white">Generic Translation</text>

      {/* Language 1 */}
      <rect
        x="450" y="520" width="120" height="60" rx="8"
        fill={getStepColor('translation').fill} stroke={getStepColor('translation').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('language1')}
      />
      <text x="510" y="555" textAnchor="middle" className="text-xs fill-white">Language 1</text>

      {/* Language 200 */}
      <rect
        x="600" y="520" width="120" height="60" rx="8"
        fill={getStepColor('translation').fill} stroke={getStepColor('translation').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('language200')}
      />
      <text x="660" y="555" textAnchor="middle" className="text-xs fill-white">Language 200</text>

      {/* NLLB & AYA 101 */}
      <rect
        x="450" y="640" width="170" height="60" rx="8"
        fill={getStepColor('agent').fill} stroke={getStepColor('agent').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('translationModels')}
      />
      <text x="535" y="675" textAnchor="middle" className="text-xs fill-white">NLLB & AYA 101</text>

      {/* Language Verification */}
      <polygon
        points="535,750 605,780 535,810 465,780"
        fill={getStepColor('decision').fill} stroke={getStepColor('decision').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('languageVerify')}
      />
      <text x="535" y="785" textAnchor="middle" className="text-xs fill-black">Language Verify</text>

      {/* Web Review Agent */}
      <rect
        x="465" y="860" width="140" height="60" rx="8"
        fill={getStepColor('agent').fill} stroke={getStepColor('agent').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('reviewAgent')}
      />
      <text x="535" y="895" textAnchor="middle" className="text-xs fill-white">Web Review Agent</text>

      {/* Yes/No Decision */}
      <polygon
        points="535,980 605,1010 535,1040 465,1010"
        fill={getStepColor('decision').fill} stroke={getStepColor('decision').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('decision')}
      />
      <text x="535" y="1015" textAnchor="middle" className="text-xs fill-black">Yes/No?</text>

      {/* Failed */}
      <ellipse
        cx="350" cy="1100" rx="60" ry="30"
        fill={getStepColor('end').fill} stroke={getStepColor('end').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('failed')}
      />
      <text x="350" y="1105" textAnchor="middle" className="text-xs fill-white">Failed</text>

      {/* Approved */}
      <ellipse
        cx="720" cy="1100" rx="60" ry="30"
        fill={getStepColor('start').fill} stroke={getStepColor('start').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('approved')}
      />
      <text x="720" y="1105" textAnchor="middle" className="text-xs fill-black">Approved</text>

      {/* Timestamp + Data - Final Step */}
      <rect
        x="465" y="1180" width="140" height="60" rx="8"
        fill={getStepColor('process').fill} stroke={getStepColor('process').stroke} strokeWidth="2"
        className="cursor-pointer hover:opacity-80"
        onClick={() => setSelectedStep('timestamp')}
      />
      <text x="535" y="1215" textAnchor="middle" className="text-xs fill-white">Timestamp + Data</text>

      {/* Flow Arrows */}
      <line x1="190" y1="80" x2="260" y2="90" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="380" y1="90" x2="500" y2="80" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="640" y1="80" x2="700" y2="80" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="840" y1="80" x2="900" y2="80" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />
      
      <line x1="320" y1="120" x2="320" y2="170" stroke="#16a34a" strokeWidth="2" markerEnd="url(#arrowGreen)" />
      <line x1="320" y1="230" x2="320" y2="280" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="390" y1="310" x2="500" y2="310" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="320" y1="340" x2="320" y2="400" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />
      
      {/* Frequency decision branches */}
      <line x1="250" y1="430" x2="190" y2="430" stroke="#dc2626" strokeWidth="2" markerEnd="url(#arrowRed)" />
      <line x1="320" y1="460" x2="320" y2="520" stroke="#16a34a" strokeWidth="2" markerEnd="url(#arrowGreen)" />
      
      <line x1="390" y1="550" x2="450" y2="550" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="570" y1="550" x2="600" y2="550" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="535" y1="580" x2="535" y2="640" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="535" y1="700" x2="535" y2="750" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="535" y1="810" x2="535" y2="860" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="535" y1="920" x2="535" y2="980" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />
      
      {/* Final decision branches */}
      <line x1="465" y1="1010" x2="410" y2="1100" stroke="#dc2626" strokeWidth="2" markerEnd="url(#arrowRed)" />
      <line x1="605" y1="1010" x2="660" y2="1100" stroke="#16a34a" strokeWidth="2" markerEnd="url(#arrowGreen)" />
      
      {/* Both Failed and Approved flow to Timestamp */}
      <line x1="350" y1="1130" x2="535" y2="1180" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="720" y1="1130" x2="535" y2="1180" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />

      {/* Labels */}
      <text x="225" y="105" className="text-xs fill-green-600">Yes</text>
      <text x="225" y="445" className="text-xs fill-red-600">Freq=1</text>
      <text x="345" y="485" className="text-xs fill-green-600">Freq&gt;2</text>
      <text x="425" y="1080" className="text-xs fill-red-600">No</text>
      <text x="635" y="1080" className="text-xs fill-green-600">Yes</text>
    </svg>
  );

  const selectedStepData = processSteps.find(step => step.id === selectedStep);

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <div className="bg-white rounded-lg border-2 border-muted" style={{ aspectRatio: '1400/1280' }}>
          {renderTerminologyValidationDiagram()}
        </div>
      </Card>

      {showExplanations && (
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="p-6">
            <h3 className="mb-4">Process Steps</h3>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {processSteps.map((step) => (
                <div
                  key={step.id}
                  className={`p-3 rounded-lg border-2 cursor-pointer transition-colors ${
                    selectedStep === step.id
                      ? 'border-primary bg-primary/5'
                      : 'border-muted hover:border-muted-foreground/50'
                  }`}
                  onClick={() => setSelectedStep(step.id)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium">{step.title}</h4>
                    <Badge variant="outline" className="text-xs">
                      {step.type}
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {step.description}
                  </p>
                </div>
              ))}
            </div>
          </Card>

          <Card className="p-6">
            <h3 className="mb-4">Step Details</h3>
            {selectedStepData ? (
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium">{selectedStepData.title}</h4>
                  <Badge variant="outline" className="mt-1">
                    {selectedStepData.type}
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground">
                  {selectedStepData.details}
                </p>
                <div className="border-t pt-4">
                  <h5 className="font-medium mb-2">Process Information:</h5>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Type: {selectedStepData.type}</li>
                    <li>• Interactive: Click to highlight</li>
                    <li>• Part of validation workflow</li>
                    <li>• Ensures terminology integrity</li>
                  </ul>
                </div>
              </div>
            ) : (
              <p className="text-muted-foreground">
                Click on any step in the diagram to see detailed information about that part of the validation process.
              </p>
            )}
          </Card>
        </div>
      )}

      <Card className="p-6">
        <h3 className="mb-4">Process Workflow Summary</h3>
        <div className="grid md:grid-cols-3 gap-4 text-sm">
          <div>
            <h4 className="font-medium mb-2">Validation Phase</h4>
            <ul className="space-y-1 text-muted-foreground">
              <li>• Term collection and source verification</li>
              <li>• Glossary and dictionary checks</li>
              <li>• Language-specific validation</li>
              <li>• Frequency analysis and filtering</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium mb-2">Translation Phase</h4>
            <ul className="space-y-1 text-muted-foreground">
              <li>• Generic translation approach</li>
              <li>• Multi-language processing (1-200)</li>
              <li>• NLLB and AYA 101 model usage</li>
              <li>• Language verification checks</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium mb-2">Review Phase</h4>
            <ul className="space-y-1 text-muted-foreground">
              <li>• Timestamp and data recording</li>
              <li>• Web Review Agent evaluation</li>
              <li>• Final approval/rejection decision</li>
              <li>• Quality assurance workflow</li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
}