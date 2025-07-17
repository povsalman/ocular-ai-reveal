
import React from 'react';
import { Clock } from 'lucide-react';
import ModuleLayout from '@/components/ModuleLayout';
import ModuleAnalysis from '@/components/ModuleAnalysis';
import { AnalysisResult } from '@/types/analysis';

const AgePrediction = () => {
  const generateMockResult = (): AnalysisResult => {
    const ageRanges = [
      { prediction: 'Estimated Age: 25-35 years', risk: 'low' as const, details: 'Young adult retinal features' },
      { prediction: 'Estimated Age: 35-45 years', risk: 'low' as const, details: 'Early adult retinal characteristics' },
      { prediction: 'Estimated Age: 45-55 years', risk: 'medium' as const, details: 'Middle-aged retinal features' },
      { prediction: 'Estimated Age: 55-65 years', risk: 'medium' as const, details: 'Mature adult retinal changes' },
      { prediction: 'Estimated Age: 65+ years', risk: 'medium' as const, details: 'Senior retinal characteristics' }
    ];
    
    const randomPrediction = ageRanges[Math.floor(Math.random() * ageRanges.length)];
    
    return {
      moduleId: 'age_prediction',
      moduleName: 'Age Prediction',
      prediction: randomPrediction.prediction,
      accuracy: 91.2 + Math.random() * 6,
      confidence: 85.4 + Math.random() * 12,
      details: randomPrediction.details,
      riskLevel: randomPrediction.risk,
      additionalInfo: 'Age prediction from retinal images leverages the correlation between retinal aging patterns and chronological age. Our model analyzes vascular architecture, optic disc changes, and other age-related retinal features to provide accurate age estimation.'
    };
  };

  return (
    <ModuleLayout
      title="Age Prediction"
      description="Biological Age Estimation from Retinal Features"
      icon={<Clock className="h-8 w-8" />}
    >
      <div className="space-y-8">
        {/* Information Section */}
        <div className="gradient-card rounded-xl p-6 medical-shadow medical-border">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">About Age Prediction</h2>
          <div className="space-y-4 text-gray-600">
            <p>
              Retinal age prediction uses the eye as a window to understand biological aging. Changes in retinal 
              structure correlate strongly with chronological age and can indicate accelerated aging patterns.
            </p>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-gray-700 mb-2">Key Features Analyzed</h3>
                <ul className="text-sm space-y-1">
                  <li>• Vascular tortuosity and caliber</li>
                  <li>• Optic disc appearance</li>
                  <li>• Macular pigmentation</li>
                  <li>• Arteriovenous ratio</li>
                </ul>
              </div>
              <div>
                <h3 className="font-semibold text-gray-700 mb-2">Training Method</h3>
                <p className="text-sm">
                  Regression-based deep learning model trained on age-labeled retinal images with 
                  cross-validation across different populations and age groups.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Analysis Section */}
        <ModuleAnalysis
          moduleId="age_prediction"
          moduleName="Age Prediction"
          generateMockResult={generateMockResult}
        />
      </div>
    </ModuleLayout>
  );
};

export default AgePrediction;
