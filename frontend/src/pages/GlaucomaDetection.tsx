
import React from 'react';
import { Stethoscope } from 'lucide-react';
import ModuleLayout from '@/components/ModuleLayout';
import ModuleAnalysis from '@/components/ModuleAnalysis';
import { AnalysisResult } from '@/types/analysis';

const GlaucomaDetection = () => {
  const generateMockResult = (): AnalysisResult => {
    const predictions = [
      { prediction: 'No Glaucoma Detected', risk: 'low' as const, details: 'Healthy optic nerve structure observed' },
      { prediction: 'Glaucoma Suspect', risk: 'medium' as const, details: 'Suspicious optic nerve changes detected' },
      { prediction: 'Early Glaucoma', risk: 'high' as const, details: 'Early glaucomatous damage present' },
      { prediction: 'Advanced Glaucoma', risk: 'high' as const, details: 'Significant glaucomatous damage detected' }
    ];
    
    const randomPrediction = predictions[Math.floor(Math.random() * predictions.length)];
    
    // Generate a mock optic cup segmentation mask
    const mockMaskImage = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZGVmcz4KICAgIDxyYWRpYWxHcmFkaWVudCBpZD0ib3B0aWNEaXNjIiBjeD0iNTAlIiBjeT0iNTAlIiByPSI1MCUiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdHlsZT0ic3RvcC1jb2xvcjojZmZmZjAwO3N0b3Atb3BhY2l0eTowLjgiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSI2MCUiIHN0eWxlPSJzdG9wLWNvbG9yOiNmZmE1MDA7c3RvcC1vcGFjaXR5OjAuNiIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0eWxlPSJzdG9wLWNvbG9yOiNmZjAwMDA7c3RvcC1vcGFjaXR5OjAuNCIvPgogICAgPC9yYWRpYWxHcmFkaWVudD4KICA8L2RlZnM+CiAgPHJlY3Qgd2lkdGg9IjMwMCIgaGVpZ2h0PSIzMDAiIGZpbGw9IiMwMDAiLz4KICA8Y2lyY2xlIGN4PSIxNTAiIGN5PSIxNTAiIHI9IjgwIiBmaWxsPSIjMzMzIiBvcGFjaXR5PSIwLjMiLz4KICA8Y2lyY2xlIGN4PSIxNTAiIGN5PSIxNTAiIHI9IjYwIiBmaWxsPSJ1cmwoI29wdGljRGlzYykiLz4KICA8Y2lyY2xlIGN4PSIxNTAiIGN5PSIxNTAiIHI9IjI1IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjgiLz4KICA8dGV4dCB4PSIxNTAiIHk9IjI4MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2ZmZiIgZm9udC1zaXplPSIxMiI+T3B0aWMgQ3VwIFNlZ21lbnRhdGlvbjwvdGV4dD4KPC9zdmc+';
    
    return {
      moduleId: 'glaucoma_detection',
      moduleName: 'Glaucoma Detection',
      prediction: randomPrediction.prediction,
      accuracy: 93.1 + Math.random() * 4,
      confidence: 88.7 + Math.random() * 9,
      details: randomPrediction.details,
      riskLevel: randomPrediction.risk,
      maskImage: mockMaskImage,
      additionalInfo: 'Our glaucoma detection model combines classification and segmentation to assess optic nerve health. It analyzes cup-to-disc ratio, neuroretinal rim characteristics, and performs precise optic cup segmentation to aid in glaucoma diagnosis and monitoring.'
    };
  };

  return (
    <ModuleLayout
      title="Glaucoma Detection"
      description="Glaucoma Detection & Optic Cup Segmentation"
      icon={<Stethoscope className="h-8 w-8" />}
    >
      <div className="space-y-8">
        {/* Information Section */}
        <div className="gradient-card rounded-xl p-6 medical-shadow medical-border">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">About Glaucoma Detection</h2>
          <div className="space-y-4 text-gray-600">
            <p>
              Glaucoma is a leading cause of blindness worldwide. Early detection through automated analysis 
              of optic disc morphology can help preserve vision through timely treatment.
            </p>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-gray-700 mb-2">Key Diagnostic Features</h3>
                <ul className="text-sm space-y-1">
                  <li>• Cup-to-disc ratio (CDR)</li>
                  <li>• Neuroretinal rim thinning</li>
                  <li>• Optic disc hemorrhages</li>
                  <li>• RNFL defects</li>
                </ul>
              </div>
              <div>
                <h3 className="font-semibold text-gray-700 mb-2">Training Method</h3>
                <p className="text-sm">
                  Multi-task learning combining classification and segmentation, trained on expert-labeled 
                  datasets with validation against clinical diagnosis and OCT measurements.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Analysis Section */}
        <ModuleAnalysis
          moduleId="glaucoma_detection"
          moduleName="Glaucoma Detection"
          generateMockResult={generateMockResult}
        />
      </div>
    </ModuleLayout>
  );
};

export default GlaucomaDetection;
