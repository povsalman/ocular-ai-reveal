import React from "react";
import { Eye } from "lucide-react";
import ModuleLayout from "@/components/ModuleLayout";
import DRModuleAnalysis from "@/components/DRModuleAnalysis";
import { AnalysisResult } from "@/types/analysis";

const DRClassification = () => {
  const generateMockResult = (): AnalysisResult => {
    const predictions = [
      {
        prediction: "No Diabetic Retinopathy",
        risk: "low" as const,
        details: "Healthy retinal structure detected",
      },
      {
        prediction: "Mild Diabetic Retinopathy",
        risk: "medium" as const,
        details: "Early stage changes observed",
      },
      {
        prediction: "Moderate Diabetic Retinopathy",
        risk: "high" as const,
        details: "Significant retinal changes detected",
      },
      {
        prediction: "Severe Diabetic Retinopathy",
        risk: "high" as const,
        details: "Advanced diabetic changes present",
      },
    ];

    const randomPrediction =
      predictions[Math.floor(Math.random() * predictions.length)];

    return {
      moduleId: "dr_classification",
      moduleName: "DR Classification",
      prediction: randomPrediction.prediction,
      accuracy: 92.3 + Math.random() * 5,
      confidence: 87.8 + Math.random() * 10,
      details: randomPrediction.details,
      riskLevel: randomPrediction.risk,
      additionalInfo:
        "Our DR classification model uses advanced convolutional neural networks trained on over 100,000 retinal images. The model can detect various stages of diabetic retinopathy with high accuracy, helping in early diagnosis and treatment planning.",
    };
  };

  return (
    <ModuleLayout
      title="DR Classification"
      description="Advanced Diabetic Retinopathy Detection & Classification"
      icon={<Eye className="h-8 w-8" />}
    >
      <div className="space-y-8">
        {/* Information Section */}
        <div className="gradient-card rounded-xl p-6 medical-shadow medical-border">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">
            About DR Classification
          </h2>
          <div className="space-y-4 text-gray-600">
            <p>
              Diabetic Retinopathy (DR) is an eye condition caused by high blood
              sugar levels damaging the blood vessels in the retina (the
              light-sensitive tissue at the back of the eye). It is a common
              complication of diabetes and can lead to vision loss or blindness
              if not detected and treated early.
            </p>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-gray-700 mb-2">
                  Training Method
                </h3>
                <p className="text-sm">
                  Vision Transformer and DenseNet201 trained on extensive
                  datasets of labeled retinal images with validation from
                  ophthalmologists.
                </p>
              </div>
              <div>
                <h3 className="font-semibold text-gray-700 mb-2">
                  Classification Stages
                </h3>
                <ul className="text-sm space-y-1 list-disc list-inside">
                  <li>
                    <strong>No DR</strong>: No signs of diabetic retinopathy.
                  </li>
                  <li>
                    <strong>Mild NPDR</strong>: Small bulges (microaneurysms)
                    form in the retinal blood vessels.
                  </li>
                  <li>
                    <strong>Moderate NPDR</strong>: Some blood vessels are
                    blocked, leading to fluid and blood leakage.
                  </li>
                  <li>
                    <strong>Severe NPDR</strong>: Many vessels are blocked; the
                    retina is deprived of oxygen (ischemia).
                  </li>
                  <li>
                    <strong>Proliferative DR</strong>: Advanced stage with
                    abnormal new blood vessels (neovascularization) that may
                    bleed, scar, or detach the retina.
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Analysis Section */}
        <DRModuleAnalysis
          moduleId="dr_classification"
          moduleName="DR Classification"
          generateMockResult={generateMockResult}
        />
      </div>
    </ModuleLayout>
  );
};

export default DRClassification;
