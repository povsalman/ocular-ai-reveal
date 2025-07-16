
import React from 'react';
import { Search } from 'lucide-react';
import ModuleLayout from '@/components/ModuleLayout';
import ModuleAnalysis from '@/components/ModuleAnalysis';
import { AnalysisResult } from '@/types/analysis';

const MyopiaDetection = () => {
  const generateMockResult = (): AnalysisResult => {
    const predictions = [
      { prediction: 'No Myopia Detected', risk: 'low' as const, details: 'Normal refractive indicators observed' },
      { prediction: 'Mild Myopia Risk', risk: 'medium' as const, details: 'Early myopic changes detected' },
      { prediction: 'Moderate Myopia Risk', risk: 'high' as const, details: 'Moderate myopic features present' },
      { prediction: 'High Myopia Risk', risk: 'high' as const, details: 'Significant myopic characteristics detected' }
    ];
    
    const randomPrediction = predictions[Math.floor(Math.random() * predictions.length)];
    
    return {
      moduleId: 'myopia_detection',
      moduleName: 'Myopia Detection',
      prediction: randomPrediction.prediction,
      accuracy: 88.9 + Math.random() * 7,
      confidence: 82.1 + Math.random() * 15,
      details: randomPrediction.details,
      riskLevel: randomPrediction.risk,
      additionalInfo: 'Myopia detection from retinal images focuses on identifying structural changes associated with myopic progression. Our model analyzes optic disc characteristics, peripapillary atrophy, and posterior pole changes to assess myopia risk and severity.'
    };
  };

  return (
    <ModuleLayout
      title="Myopia Detection"
      description="Myopia Risk Assessment & Detection"
      icon={<Search className="h-8 w-8 text-blue-700" />}
      className="bg-gradient-to-br from-green-100 to-green-50 min-h-screen"
    >
      <div className="space-y-8">
        {/* Information Section */}
        <div className="gradient-card rounded-xl p-6 medical-shadow medical-border bg-green-50 border-green-200">
          <h2 className="text-2xl font-bold text-black mb-4">About Myopia Detection</h2>
          <div className="space-y-4 text-black">
            <p>
              Myopia (nearsightedness) is increasingly prevalent worldwide. Early detection through retinal analysis 
              can help identify individuals at risk and enable timely intervention to slow progression.
            </p>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-black mb-2">Myopic Changes Detected</h3>
                <ul className="text-sm space-y-1">
                  <li>• Peripapillary atrophy</li>
                  <li>• Tilted optic disc</li>
                  <li>• Posterior staphyloma</li>
                  <li>• Tessellated fundus</li>
                </ul>
              </div>
              <div>
                <h3 className="font-semibold text-black mb-2">Training Method</h3>
                <p className="text-sm">
                  Classification model trained on diverse datasets with myopia severity labels, 
                  validated against clinical refractive measurements and expert annotations.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Analysis Section */}
        <ModuleAnalysis
          moduleId="myopia_detection"
          moduleName="Myopia Detection"
          generateMockResult={generateMockResult}
        />
      </div>
    </ModuleLayout>
  );
};

export default MyopiaDetection;
