import { useState } from 'react';
import { Card } from './ui/card';
import { Badge } from './ui/badge';

interface DiagramViewerProps {
  diagramType: string;
  showExplanations: boolean;
}

interface DiagramElement {
  id: string;
  title: string;
  description: string;
  type: 'process' | 'decision' | 'data' | 'connection' | 'start' | 'end';
}

const diagramElements: Record<string, DiagramElement[]> = {
  flowchart: [
    { id: 'start', title: 'Start', description: 'Beginning of the process flow', type: 'start' },
    { id: 'input', title: 'User Input', description: 'Collect user data and requirements', type: 'process' },
    { id: 'validate', title: 'Validate Data', description: 'Check if input data meets requirements', type: 'decision' },
    { id: 'process', title: 'Process Data', description: 'Transform and analyze the input data', type: 'process' },
    { id: 'output', title: 'Generate Output', description: 'Create the final result or response', type: 'process' },
    { id: 'end', title: 'End', description: 'Process completion', type: 'end' }
  ],
  system: [
    { id: 'client', title: 'Client Layer', description: 'User interface and presentation layer', type: 'process' },
    { id: 'api', title: 'API Gateway', description: 'Routes requests and handles authentication', type: 'process' },
    { id: 'service', title: 'Business Logic', description: 'Core application logic and rules', type: 'process' },
    { id: 'database', title: 'Database', description: 'Data storage and persistence layer', type: 'data' }
  ],
  workflow: [
    { id: 'init', title: 'Initialize', description: 'Set up workflow environment and parameters', type: 'start' },
    { id: 'task1', title: 'Task A', description: 'First parallel task in the workflow', type: 'process' },
    { id: 'task2', title: 'Task B', description: 'Second parallel task in the workflow', type: 'process' },
    { id: 'merge', title: 'Merge Results', description: 'Combine outputs from parallel tasks', type: 'process' },
    { id: 'complete', title: 'Complete', description: 'Workflow completion', type: 'end' }
  ],
  network: [
    { id: 'router', title: 'Router', description: 'Central routing device managing network traffic', type: 'process' },
    { id: 'switch1', title: 'Switch A', description: 'Network switch for subnet A', type: 'process' },
    { id: 'switch2', title: 'Switch B', description: 'Network switch for subnet B', type: 'process' },
    { id: 'server', title: 'Server', description: 'Main application server', type: 'data' },
    { id: 'clients', title: 'Client Devices', description: 'End user devices and workstations', type: 'process' }
  ]
};

export function DiagramViewer({ diagramType, showExplanations }: DiagramViewerProps) {
  const [selectedElement, setSelectedElement] = useState<string | null>(null);
  const elements = diagramElements[diagramType] || [];

  const renderFlowchart = () => (
    <svg viewBox="0 0 800 600" className="w-full h-full">
      {/* Start */}
      <ellipse
        cx="400" cy="80" rx="60" ry="30"
        fill="#4ade80" stroke="#166534" strokeWidth="2"
        className="cursor-pointer hover:fill-green-300"
        onClick={() => setSelectedElement('start')}
      />
      <text x="400" y="85" textAnchor="middle" className="text-sm font-medium">Start</text>

      {/* User Input */}
      <rect
        x="330" y="140" width="140" height="60" rx="5"
        fill="#60a5fa" stroke="#1d4ed8" strokeWidth="2"
        className="cursor-pointer hover:fill-blue-300"
        onClick={() => setSelectedElement('input')}
      />
      <text x="400" y="175" textAnchor="middle" className="text-sm font-medium">User Input</text>

      {/* Validate Data */}
      <polygon
        points="400,230 460,270 400,310 340,270"
        fill="#fbbf24" stroke="#d97706" strokeWidth="2"
        className="cursor-pointer hover:fill-yellow-300"
        onClick={() => setSelectedElement('validate')}
      />
      <text x="400" y="275" textAnchor="middle" className="text-sm font-medium">Validate?</text>

      {/* Process Data */}
      <rect
        x="330" y="340" width="140" height="60" rx="5"
        fill="#a78bfa" stroke="#7c3aed" strokeWidth="2"
        className="cursor-pointer hover:fill-purple-300"
        onClick={() => setSelectedElement('process')}
      />
      <text x="400" y="375" textAnchor="middle" className="text-sm font-medium">Process Data</text>

      {/* Generate Output */}
      <rect
        x="330" y="430" width="140" height="60" rx="5"
        fill="#fb7185" stroke="#e11d48" strokeWidth="2"
        className="cursor-pointer hover:fill-pink-300"
        onClick={() => setSelectedElement('output')}
      />
      <text x="400" y="465" textAnchor="middle" className="text-sm font-medium">Generate Output</text>

      {/* End */}
      <ellipse
        cx="400" cy="540" rx="60" ry="30"
        fill="#f87171" stroke="#dc2626" strokeWidth="2"
        className="cursor-pointer hover:fill-red-300"
        onClick={() => setSelectedElement('end')}
      />
      <text x="400" y="545" textAnchor="middle" className="text-sm font-medium">End</text>

      {/* Arrows */}
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 10 3, 0 6" fill="#374151" />
        </marker>
      </defs>

      <line x1="400" y1="110" x2="400" y2="140" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="400" y1="200" x2="400" y2="230" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="400" y1="310" x2="400" y2="340" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="400" y1="400" x2="400" y2="430" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="400" y1="490" x2="400" y2="510" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow)" />

      {/* Decision arrows */}
      <line x1="460" y1="270" x2="550" y2="270" stroke="#dc2626" strokeWidth="2" markerEnd="url(#arrow)" />
      <text x="520" y="265" className="text-xs fill-red-600">No</text>
      <line x1="550" y1="270" x2="550" y2="175" stroke="#dc2626" strokeWidth="2" />
      <line x1="550" y1="175" x2="470" y2="175" stroke="#dc2626" strokeWidth="2" markerEnd="url(#arrow)" />

      <text x="410" y="325" className="text-xs fill-green-600">Yes</text>
    </svg>
  );

  const renderSystemArchitecture = () => (
    <svg viewBox="0 0 800 600" className="w-full h-full">
      {/* Client Layer */}
      <rect
        x="50" y="100" width="150" height="80" rx="10"
        fill="#ddd6fe" stroke="#7c3aed" strokeWidth="2"
        className="cursor-pointer hover:fill-purple-200"
        onClick={() => setSelectedElement('client')}
      />
      <text x="125" y="145" textAnchor="middle" className="text-sm font-medium">Client Layer</text>

      {/* API Gateway */}
      <rect
        x="300" y="100" width="150" height="80" rx="10"
        fill="#fef3c7" stroke="#d97706" strokeWidth="2"
        className="cursor-pointer hover:fill-yellow-200"
        onClick={() => setSelectedElement('api')}
      />
      <text x="375" y="145" textAnchor="middle" className="text-sm font-medium">API Gateway</text>

      {/* Business Logic */}
      <rect
        x="550" y="100" width="150" height="80" rx="10"
        fill="#dcfce7" stroke="#16a34a" strokeWidth="2"
        className="cursor-pointer hover:fill-green-200"
        onClick={() => setSelectedElement('service')}
      />
      <text x="625" y="145" textAnchor="middle" className="text-sm font-medium">Business Logic</text>

      {/* Database */}
      <rect
        x="550" y="300" width="150" height="80" rx="10"
        fill="#fed7e2" stroke="#e11d48" strokeWidth="2"
        className="cursor-pointer hover:fill-pink-200"
        onClick={() => setSelectedElement('database')}
      />
      <text x="625" y="345" textAnchor="middle" className="text-sm font-medium">Database</text>

      {/* Arrows */}
      <defs>
        <marker id="arrow2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 10 3, 0 6" fill="#374151" />
        </marker>
      </defs>

      <line x1="200" y1="140" x2="300" y2="140" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow2)" />
      <line x1="450" y1="140" x2="550" y2="140" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow2)" />
      <line x1="625" y1="180" x2="625" y2="300" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow2)" />

      {/* Labels */}
      <text x="250" y="130" className="text-xs">HTTP/API</text>
      <text x="500" y="130" className="text-xs">Service Call</text>
      <text x="640" y="240" className="text-xs">Query</text>
    </svg>
  );

  const renderWorkflow = () => (
    <svg viewBox="0 0 800 600" className="w-full h-full">
      {/* Initialize */}
      <ellipse
        cx="150" cy="140" rx="60" ry="30"
        fill="#4ade80" stroke="#166534" strokeWidth="2"
        className="cursor-pointer hover:fill-green-300"
        onClick={() => setSelectedElement('init')}
      />
      <text x="150" y="145" textAnchor="middle" className="text-sm font-medium">Initialize</text>

      {/* Task A */}
      <rect
        x="300" y="80" width="120" height="60" rx="5"
        fill="#60a5fa" stroke="#1d4ed8" strokeWidth="2"
        className="cursor-pointer hover:fill-blue-300"
        onClick={() => setSelectedElement('task1')}
      />
      <text x="360" y="115" textAnchor="middle" className="text-sm font-medium">Task A</text>

      {/* Task B */}
      <rect
        x="300" y="180" width="120" height="60" rx="5"
        fill="#a78bfa" stroke="#7c3aed" strokeWidth="2"
        className="cursor-pointer hover:fill-purple-300"
        onClick={() => setSelectedElement('task2')}
      />
      <text x="360" y="215" textAnchor="middle" className="text-sm font-medium">Task B</text>

      {/* Merge Results */}
      <rect
        x="500" y="130" width="120" height="60" rx="5"
        fill="#fbbf24" stroke="#d97706" strokeWidth="2"
        className="cursor-pointer hover:fill-yellow-300"
        onClick={() => setSelectedElement('merge')}
      />
      <text x="560" y="165" textAnchor="middle" className="text-sm font-medium">Merge Results</text>

      {/* Complete */}
      <ellipse
        cx="700" cy="160" rx="60" ry="30"
        fill="#f87171" stroke="#dc2626" strokeWidth="2"
        className="cursor-pointer hover:fill-red-300"
        onClick={() => setSelectedElement('complete')}
      />
      <text x="700" y="165" textAnchor="middle" className="text-sm font-medium">Complete</text>

      {/* Arrows */}
      <defs>
        <marker id="arrow3" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 10 3, 0 6" fill="#374151" />
        </marker>
      </defs>

      <line x1="210" y1="130" x2="300" y2="110" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow3)" />
      <line x1="210" y1="150" x2="300" y2="210" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow3)" />
      <line x1="420" y1="110" x2="500" y2="150" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow3)" />
      <line x1="420" y1="210" x2="500" y2="170" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow3)" />
      <line x1="620" y1="160" x2="640" y2="160" stroke="#374151" strokeWidth="2" markerEnd="url(#arrow3)" />

      {/* Parallel indicator */}
      <text x="240" y="100" className="text-xs fill-blue-600">Parallel</text>
      <text x="240" y="230" className="text-xs fill-purple-600">Execution</text>
    </svg>
  );

  const renderNetwork = () => (
    <svg viewBox="0 0 800 600" className="w-full h-full">
      {/* Router */}
      <rect
        x="350" y="80" width="100" height="60" rx="10"
        fill="#fbbf24" stroke="#d97706" strokeWidth="2"
        className="cursor-pointer hover:fill-yellow-300"
        onClick={() => setSelectedElement('router')}
      />
      <text x="400" y="115" textAnchor="middle" className="text-sm font-medium">Router</text>

      {/* Switch A */}
      <rect
        x="150" y="200" width="100" height="60" rx="10"
        fill="#60a5fa" stroke="#1d4ed8" strokeWidth="2"
        className="cursor-pointer hover:fill-blue-300"
        onClick={() => setSelectedElement('switch1')}
      />
      <text x="200" y="235" textAnchor="middle" className="text-sm font-medium">Switch A</text>

      {/* Switch B */}
      <rect
        x="550" y="200" width="100" height="60" rx="10"
        fill="#a78bfa" stroke="#7c3aed" strokeWidth="2"
        className="cursor-pointer hover:fill-purple-300"
        onClick={() => setSelectedElement('switch2')}
      />
      <text x="600" y="235" textAnchor="middle" className="text-sm font-medium">Switch B</text>

      {/* Server */}
      <rect
        x="350" y="320" width="100" height="80" rx="10"
        fill="#fed7e2" stroke="#e11d48" strokeWidth="2"
        className="cursor-pointer hover:fill-pink-200"
        onClick={() => setSelectedElement('server')}
      />
      <text x="400" y="365" textAnchor="middle" className="text-sm font-medium">Server</text>

      {/* Client Devices */}
      <circle
        cx="100" cy="350" r="40"
        fill="#dcfce7" stroke="#16a34a" strokeWidth="2"
        className="cursor-pointer hover:fill-green-200"
        onClick={() => setSelectedElement('clients')}
      />
      <text x="100" y="355" textAnchor="middle" className="text-sm font-medium">Clients</text>

      <circle
        cx="700" cy="350" r="40"
        fill="#dcfce7" stroke="#16a34a" strokeWidth="2"
        className="cursor-pointer hover:fill-green-200"
        onClick={() => setSelectedElement('clients')}
      />
      <text x="700" y="355" textAnchor="middle" className="text-sm font-medium">Clients</text>

      {/* Connections */}
      <defs>
        <marker id="arrow4" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 10 3, 0 6" fill="#374151" />
        </marker>
      </defs>

      <line x1="380" y1="140" x2="220" y2="200" stroke="#374151" strokeWidth="2" />
      <line x1="420" y1="140" x2="580" y2="200" stroke="#374151" strokeWidth="2" />
      <line x1="400" y1="140" x2="400" y2="320" stroke="#374151" strokeWidth="2" />
      <line x1="150" y1="230" x2="140" y2="350" stroke="#374151" strokeWidth="2" />
      <line x1="650" y1="230" x2="660" y2="350" stroke="#374151" strokeWidth="2" />

      {/* Network Labels */}
      <text x="290" y="170" className="text-xs">Subnet A</text>
      <text x="510" y="170" className="text-xs">Subnet B</text>
      <text x="420" y="240" className="text-xs">Backbone</text>
    </svg>
  );

  const renderDiagram = () => {
    switch (diagramType) {
      case 'flowchart':
        return renderFlowchart();
      case 'system':
        return renderSystemArchitecture();
      case 'workflow':
        return renderWorkflow();
      case 'network':
        return renderNetwork();
      default:
        return renderFlowchart();
    }
  };

  const selectedElementData = elements.find(el => el.id === selectedElement);

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <div className="aspect-video bg-white rounded-lg border-2 border-muted">
          {renderDiagram()}
        </div>
      </Card>

      {showExplanations && (
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="p-6">
            <h3 className="mb-4">Diagram Elements</h3>
            <div className="space-y-3">
              {elements.map((element) => (
                <div
                  key={element.id}
                  className={`p-3 rounded-lg border-2 cursor-pointer transition-colors ${
                    selectedElement === element.id
                      ? 'border-primary bg-primary/5'
                      : 'border-muted hover:border-muted-foreground/50'
                  }`}
                  onClick={() => setSelectedElement(element.id)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium">{element.title}</h4>
                    <Badge variant="outline" className="text-xs">
                      {element.type}
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {element.description}
                  </p>
                </div>
              ))}
            </div>
          </Card>

          <Card className="p-6">
            <h3 className="mb-4">Selected Element Details</h3>
            {selectedElementData ? (
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium">{selectedElementData.title}</h4>
                  <Badge variant="outline" className="mt-1">
                    {selectedElementData.type}
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground">
                  {selectedElementData.description}
                </p>
                <div className="border-t pt-4">
                  <h5 className="font-medium mb-2">Element Properties:</h5>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Type: {selectedElementData.type}</li>
                    <li>• Interactive: Yes</li>
                    <li>• SVG-based rendering</li>
                    <li>• Click to select and highlight</li>
                  </ul>
                </div>
              </div>
            ) : (
              <p className="text-muted-foreground">
                Click on any element in the diagram to see detailed information.
              </p>
            )}
          </Card>
        </div>
      )}

      <Card className="p-6">
        <h3 className="mb-4">Diagram Features</h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div>
            <h4 className="font-medium mb-2">Interactive Elements</h4>
            <ul className="space-y-1 text-muted-foreground">
              <li>• Click elements for detailed explanations</li>
              <li>• Hover effects for better UX</li>
              <li>• Color-coded element types</li>
              <li>• Responsive SVG scaling</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium mb-2">Technical Details</h4>
            <ul className="space-y-1 text-muted-foreground">
              <li>• Pure SVG implementation</li>
              <li>• Scalable vector graphics</li>
              <li>• Accessible markup</li>
              <li>• Export-ready format</li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
}