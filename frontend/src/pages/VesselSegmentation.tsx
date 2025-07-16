
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
            <br></br>
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
                  The system automatically selects the best performing model based on <b>confidence score (entropy-based)</b> from DRIVE, CHASEDB1, HRF, and STARE. The model with the highest confidence (lowest average entropy) is used for segmentation, ensuring optimal results for each input image.
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

        {/* Dataset Performance Section */}{/* Applications Section */}
        <div className="gradient-card rounded-xl p-6 medical-shadow medical-border mt-6">
          <h3 className="text-xl font-semibold text-gray-800 mb-3">Applications of Retinal Vessel Segmentation</h3>
          <p className="text-base text-gray-600 leading-relaxed mb-2">
            Retinal vessel segmentation plays a critical role in diagnosing and monitoring diseases such as diabetic retinopathy, hypertension, and arteriosclerosis. It enables measurement of vessel width, length, branching angles, and tortuosity which are vital indicators in ophthalmology and cardiovascular screening. These masks assist in:
          </p>
          <ul className="list-disc pl-6 text-base text-gray-600 space-y-1">
            <li>Automated screening for diabetic and hypertensive retinopathy</li>
            <li>Quantitative vascular analysis (diameter, tortuosity)</li>
            <li>Image registration and retinal map generation</li>  
            <li>Guiding laser treatments through precise vascular overlays</li>
          </ul>
          <br></br>
          <p className="text-base text-gray-600 leading-relaxed mb-2">You can find more information about the applications of retinal vessel segmentation <a href="https://drive.grand-challenge.org/" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">here</a>.</p>
        </div>

      </div>
    </ModuleLayout>
  );
};

export default VesselSegmentation;
