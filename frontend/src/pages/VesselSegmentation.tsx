
import React from 'react';
import { Brain } from 'lucide-react';
import ModuleLayout from '@/components/ModuleLayout';
import ModuleAnalysis from '@/components/ModuleAnalysis';
import { AnalysisResult } from '@/types/analysis';

const VesselSegmentation = () => {
  const generateMockResult = (): AnalysisResult => {
    const predictions = [
      { prediction: 'Normal Vessel Architecture', risk: 'low' as const, details: 'Healthy vascular pattern detected' },
      { prediction: 'Mild Vessel Irregularities', risk: 'medium' as const, details: 'Minor vascular changes noted' },
      { prediction: 'Significant Vessel Changes', risk: 'high' as const, details: 'Notable vascular abnormalities detected' }
    ];
    
    const randomPrediction = predictions[Math.floor(Math.random() * predictions.length)];
    
    // Generate a mock segmentation mask (in practice, this would be the actual AI output)
    const mockMaskImage = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZGVmcz4KICAgIDxwYXR0ZXJuIGlkPSJ2ZXNzZWxzIiBwYXR0ZXJuVW5pdHM9InVzZXJTcGFjZU9uVXNlIiB3aWR0aD0iMzAwIiBoZWlnaHQ9IjMwMCI+CiAgICAgIDxyZWN0IHdpZHRoPSIzMDAiIGhlaWdodD0iMzAwIiBmaWxsPSIjMDAwIi8+CiAgICAgIDxjaXJjbGUgY3g9IjE1MCIgY3k9IjE1MCIgcj0iMTAwIiBmaWxsPSIjMzMzIiBvcGFjaXR5PSIwLjMiLz4KICAgICAgPHBhdGggZD0iTTUwIDEwMCBRMTUwIDUwIDI1MCAyMDAiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIzIiBmaWxsPSJub25lIi8+CiAgICAgIDxwYXRoIGQ9Ik0xMDAgNTAgUTIwMCAxNTAgMTAwIDI1MCIgc3Ryb2tlPSIjZmZmIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9Im5vbmUiLz4KICAgICAgPHBhdGggZD0iTTIwMCA4MCBRMTU0IDE2MCAyMDAgeHoMIiBzdHJva2U9IiNmZmYiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0ibm9uZSIvPgogICAgPC9wYXR0ZXJuPgogIDwvZGVmcz4KICA8cmVjdCB3aWR0aD0iMzAwIiBoZWlnaHQ9IjMwMCIgZmlsbD0idXJsKCN2ZXNzZWxzKSIvPgogIDx0ZXh0IHg9IjE1MCIgeT0iMjgwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjZmZmIiBmb250LXNpemU9IjEyIj5TZWdtZW50ZWQgVmVzc2VsczwvdGV4dD4KPC9zdmc+';
    
    return {
      moduleId: 'vessel_segmentation',
      moduleName: 'Vessel Segmentation',
      prediction: randomPrediction.prediction,
      accuracy: 94.7 + Math.random() * 3,
      confidence: 89.2 + Math.random() * 8,
      details: randomPrediction.details,
      riskLevel: randomPrediction.risk,
      maskImage: mockMaskImage,
      additionalInfo: 'Our vessel segmentation model uses the R2UNet architecture, specifically designed for biomedical image segmentation. This model provides precise pixel-level segmentation of retinal blood vessels, enabling detailed vascular analysis and pathology detection.'
    };
  };

  return (
    <ModuleLayout
      title="Vessel Segmentation"
      description="Retinal Blood Vessel Segmentation using R2UNet"
      icon={<Brain className="h-8 w-8" />}
    >
      <div className="space-y-8">
        {/* Information Section */}
        <div className="gradient-card rounded-xl p-6 medical-shadow medical-border">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">About Vessel Segmentation</h2>
          <div className="space-y-4 text-gray-600">
            <p>
              Retinal vessel segmentation is crucial for diagnosing various eye diseases. Our R2UNet-based model provides 
              accurate pixel-level segmentation of blood vessels in retinal fundus images.
            </p>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-gray-700 mb-2">R2UNet Architecture</h3>
                <p className="text-sm">
                  R2UNet combines the power of U-Net with recurrent and residual connections, providing superior 
                  segmentation performance for thin vessel structures.
                </p>
              </div>
              <div>
                <h3 className="font-semibold text-gray-700 mb-2">Training Method</h3>
                <p className="text-sm">
                  Trained on multiple public datasets (DRIVE, STARE, CHASE_DB1) with data augmentation techniques 
                  and expert annotations for ground truth validation.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Analysis Section */}
        <ModuleAnalysis
          moduleId="vessel_segmentation"
          moduleName="Vessel Segmentation"
          generateMockResult={generateMockResult}
        />
      </div>
    </ModuleLayout>
  );
};

export default VesselSegmentation;
