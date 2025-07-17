
import React from 'react';
import { Search } from 'lucide-react';
import ModuleLayout from '@/components/ModuleLayout';
import ModuleAnalysis from '@/components/ModuleAnalysis';
import { AnalysisResult } from '@/types/analysis';

const generateFeatureSummary = (features?: Record<string, number | string>): string => {
  if (!features || Object.keys(features).length === 0) {
    return 'No specific features were extracted for this analysis.';
  }
  let summary = '**Feature-based Clinical Summary:**\n';
  if (features.avg_vessel_width !== undefined) {
    summary += `- **Average Vessel Width:** ${features.avg_vessel_width} pixels. Wider vessels may be associated with vascular remodeling in myopia.\n`;
  }
  if (features.vessel_width_std !== undefined) {
    summary += `- **Vessel Width Std Dev:** ${features.vessel_width_std} pixels. Greater variability may indicate irregular vessel structure.\n`;
  }
  if (features.max_vessel_width !== undefined) {
    summary += `- **Maximum Vessel Width:** ${features.max_vessel_width} pixels. High values may reflect dilated vessels.\n`;
  }
  if (features.optic_disc_area_ratio !== undefined) {
    summary += `- **Optic Disc Area Ratio:** ${features.optic_disc_area_ratio}. Larger ratios can be seen in myopic eyes.\n`;
  }
  if (features.disc_circularity !== undefined) {
    summary += `- **Disc Circularity:** ${features.disc_circularity}. Lower values may indicate a more oval or tilted disc, common in myopia.\n`;
  }
  if (features.disc_displacement !== undefined) {
    summary += `- **Disc Displacement:** ${features.disc_displacement} pixels. Greater displacement may be associated with myopic changes.\n`;
  }
  if (features.disc_eccentricity !== undefined) {
    summary += `- **Disc Eccentricity:** ${features.disc_eccentricity}. Higher eccentricity suggests a more elongated optic disc.\n`;
  }
  if (features.cup_to_disc_ratio !== undefined) {
    summary += `- **Cup-to-Disc Ratio:** ${features.cup_to_disc_ratio}. Used to assess optic nerve health; higher ratios may indicate risk of glaucoma.\n`;
  }
  if (features.disc_aspect_ratio !== undefined) {
    summary += `- **Disc Aspect Ratio:** ${features.disc_aspect_ratio}. Reflects the shape of the optic disc.\n`;
  }
  if (features.disc_ellipse_angle !== undefined) {
    summary += `- **Disc Ellipse Angle:** ${features.disc_ellipse_angle}°. Orientation of the optic disc ellipse.\n`;
  }
  if (features.macular_mean_intensity_r0 !== undefined) {
    summary += `- **Macular Mean Intensity (r0):** ${features.macular_mean_intensity_r0}. Reflects brightness in the macular region.\n`;
  }
  return summary.trim();
};

const MyopiaDetection = () => {
  const generateMockResult = (): AnalysisResult => {
    const predictions = [
      { prediction: 'No Myopia Detected', risk: 'low' as const, details: 'Normal refractive indicators observed' },
      { prediction: 'Mild Myopia Risk', risk: 'medium' as const, details: 'Early myopic changes detected' },
      { prediction: 'Moderate Myopia Risk', risk: 'high' as const, details: 'Moderate myopic features present' },
      { prediction: 'High Myopia Risk', risk: 'high' as const, details: 'Significant myopic characteristics detected' }
    ];
    
    const randomPrediction = predictions[Math.floor(Math.random() * predictions.length)];

    // Mock extracted features for demonstration
    const mockFeatures = {
      axial_length: 26.7,
      disc_tilt_ratio: 1.4,
      peripapillary_atrophy_area: 2.1,
      posterior_staphyloma: 'Present',
      tessellated_fundus: 'Marked',
      spherical_equivalent: -5.25,
      choroidal_thickness: 120,
    };

    return {
      moduleId: 'myopia_detection',
      moduleName: 'Myopia Detection',
      prediction: randomPrediction.prediction,
      accuracy: 88.9 + Math.random() * 7,
      confidence: 82.1 + Math.random() * 15,
      details: randomPrediction.details,
      riskLevel: randomPrediction.risk,
      features: mockFeatures,
      additionalInfo: generateFeatureSummary(mockFeatures) + '\n\nMyopia detection from retinal images focuses on identifying structural changes associated with myopic progression. Our model analyzes optic disc characteristics, peripapillary atrophy, and posterior pole changes to assess myopia risk and severity.'
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
export { generateFeatureSummary };
