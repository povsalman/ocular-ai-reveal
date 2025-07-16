
import React from 'react';
import { Brain } from 'lucide-react';
import ModuleLayout from '@/components/ModuleLayout';
import ModuleAnalysis from '@/components/ModuleAnalysis';
import { AnalysisResult } from '@/types/analysis';

const VesselSegmentation = () => {
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
              Retinal vessel segmentation is crucial for diagnosing various eye diseases. Our multi-dataset R2UNet-based model provides 
              accurate pixel-level segmentation of blood vessels in retinal fundus images using the best performing model from DRIVE, CHASEDB1, HRF, and STARE datasets.
            </p>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-gray-700 mb-2">Multi-Dataset R2UNet Architecture</h3>
                <p className="text-sm">
                  R2UNet combines the power of U-Net with recurrent and residual connections, providing superior 
                  segmentation performance for thin vessel structures across multiple datasets.
                </p>
              </div>
              <div>
                <h3 className="font-semibold text-gray-700 mb-2">Intelligent Model Selection</h3>
                <p className="text-sm">
                  The system automatically selects the best performing model based on Dice Coefficient, with Sensitivity and Specificity as tie-breakers, 
                  ensuring optimal results for each input image.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Analysis Section */}
        <ModuleAnalysis
          moduleId="vessel_segmentation"
          moduleName="Vessel Segmentation"
        />
      </div>
    </ModuleLayout>
  );
};

export default VesselSegmentation;
