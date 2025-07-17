import React, { useState } from "react";
import { Brain, AlertTriangle, CheckCircle, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import ImageUpload from "@/components/ImageUpload";
import { AnalysisResult as AnalysisResultType } from "@/types/analysis";

interface ModuleAnalysisProps {
  moduleId: string;
  moduleName: string;
  generateMockResult: (moduleId: string) => AnalysisResultType;
}

interface APIResponse {
  status: string;
  predicted_class: string;
  confidence: number;
  model_used: string;
  gradcam_image?: string;
}

const DR_STAGE_DESCRIPTIONS: Record<string, string> = {
  "No DR": "No signs of diabetic retinopathy were detected.",
  "Mild NPDR":
    "Early signs such as microaneurysms are present. Regular monitoring is recommended.",
  "Moderate NPDR":
    "More extensive damage is present, including blood vessel changes. Closer monitoring and treatment are required.",
  "Severe NPDR":
    "Significant damage with blocked blood vessels. Urgent treatment may be necessary.",
  "Proliferative DR":
    "The most advanced stage with new abnormal blood vessels. Immediate medical attention is crucial.",
};

const DRModuleAnalysis: React.FC<ModuleAnalysisProps> = ({
  moduleId,
  moduleName,
  generateMockResult,
}) => {
  const [selectedImage, setSelectedImage] = useState<string>("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] =
    useState<AnalysisResultType | null>(null);
  const { toast } = useToast();

  const handleImageSelect = (file: File, preview: string) => {
    setSelectedFile(file);
    setSelectedImage(preview);
    setAnalysisResult(null);
    console.log("Image selected:", file.name, file.type);
  };

  const handleRemoveImage = () => {
    setSelectedImage("");
    setSelectedFile(null);
    setAnalysisResult(null);
  };

  const mapRiskLevel = (confidence: number): "low" | "medium" | "high" => {
    if (confidence >= 80) return "low";
    if (confidence >= 60) return "medium";
    return "high";
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 80) return "text-green-600";
    if (confidence >= 60) return "text-yellow-600";
    return "text-red-600";
  };

  const getConfidenceIcon = (confidence: number) => {
    if (confidence >= 80)
      return <CheckCircle className="h-5 w-5 text-green-600" />;
    if (confidence >= 60)
      return <AlertTriangle className="h-5 w-5 text-yellow-600" />;
    return <XCircle className="h-5 w-5 text-red-600" />;
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      toast({
        title: "Missing Image",
        description: "Please select an image to analyze.",
        variant: "destructive",
      });
      return;
    }

    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("model_type", "dr"); // Only DR Classification

      const response = await fetch("http://localhost:8000/predict/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok)
        throw new Error(`HTTP error! status: ${response.status}`);
      const apiResult: APIResponse = await response.json();

      if (apiResult.status !== "success") throw new Error(`Prediction failed`);

      const result: AnalysisResultType = {
        moduleId,
        moduleName,
        prediction: apiResult.predicted_class,
        confidence: apiResult.confidence * 100,
        riskLevel: mapRiskLevel(apiResult.confidence * 100),
        gradcam_image: apiResult.gradcam_image || "",
        additionalInfo: `Model used: ${apiResult.model_used}`,
        accuracy: 0, // Placeholder if accuracy not returned
      };

      setAnalysisResult(result);

      toast({
        title: "Analysis Complete",
        description: `${moduleName} analysis finished successfully.`,
      });
    } catch (error) {
      console.error("Analysis error:", error);
      const mockResult = generateMockResult(moduleId);
      setAnalysisResult(mockResult);

      toast({
        title: "Analysis Complete (Mock)",
        description: `${moduleName} analysis completed using mock data due to API error.`,
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="gradient-card rounded-xl p-6 medical-shadow medical-border">
        <ImageUpload
          onImageSelect={handleImageSelect}
          selectedImage={selectedImage}
          onRemoveImage={handleRemoveImage}
          isAnalyzing={isAnalyzing}
        />
      </div>

      <div className="text-center">
        <Button
          onClick={handleAnalyze}
          disabled={!selectedFile || isAnalyzing}
          className="gradient-button text-white px-8 py-3 text-lg font-semibold rounded-xl"
        >
          {isAnalyzing ? (
            <div className="flex items-center space-x-2">
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></div>
              <span>Analyzing...</span>
            </div>
          ) : (
            <div className="flex items-center space-x-2">
              <Brain className="h-5 w-5" />
              <span>Start Analysis</span>
            </div>
          )}
        </Button>
      </div>

      {analysisResult ? (
        <div className="space-y-6">
          <div className="grid lg:grid-cols-2 gap-6">
            {/* Uploaded image */}
            <div className="gradient-card rounded-xl p-6 medical-shadow medical-border">
              <h3 className="text-lg font-semibold text-gray-700 mb-4">
                Original Image
              </h3>
              {selectedImage && (
                <>
                  <div className="w-[480px] h-[480px] mx-auto bg-gray-50 rounded-lg flex items-center justify-center p-4">
                    <img
                      src={selectedImage}
                      alt="Uploaded retinal image"
                      className="w-[480px] h-[480px] rounded-lg shadow-sm"
                    />
                  </div>
                  <p className="mt-4 text-sm text-gray-600 text-center leading-relaxed">
                    This is the uploaded <strong>retinal fundus image</strong>.
                    It serves as the input for the AI model, which analyzes
                    visible blood vessels, microaneurysms, and other patterns to
                    assess the presence and stage of diabetic retinopathy.
                  </p>
                </>
              )}
            </div>

            {/* Grad-CAM */}
            <div className="gradient-card rounded-xl p-6 medical-shadow medical-border">
              <h3 className="text-lg font-semibold text-gray-700 mb-4">
                Grad-CAM Heatmap
              </h3>
              {analysisResult.gradcam_image ? (
                <>
                  <div className="w-[480px] h-[480px] mx-auto bg-gray-50 rounded-lg flex items-center justify-center p-4">
                    <img
                      src={`data:image/png;base64,${analysisResult.gradcam_image}`}
                      alt="Grad-CAM"
                      className="w-[480px] h-[480px] rounded-lg shadow-sm"
                    />
                  </div>

                  {/* Grad-CAM Explanation */}
                  <div className="mt-4 text-sm text-gray-600 text-center leading-relaxed">
                    
                    This Grad-CAM visualization highlights the regions in the
                    retinal image that the model focused on to make its
                    prediction.
                    <br />
                    <span className="text-blue-700 font-medium">
                      Brighter red or yellow areas
                    </span>{" "}
                    indicate regions the model found more important, while{" "}
                    <span className="text-gray-500 font-medium">
                      cooler or darker regions
                    </span>{" "}
                    had less influence.
                  </div>
                </>
              ) : (
                <div className="w-[360px] h-[360px] bg-gray-100 rounded-lg flex items-center justify-center">
                  <XCircle className="h-12 w-12 text-gray-400" />
                  <span className="ml-2 text-gray-500">
                    No Grad-CAM available
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Prediction Result */}
          <div className="gradient-card rounded-xl p-6 medical-shadow medical-border text-center">
            <h3 className="text-lg font-semibold text-gray-700 mb-2">
              Prediction
            </h3>
            <p className="text-2xl font-bold text-gray-800">
              {analysisResult.prediction}
            </p>
            {/* Description */}
            <p className="mt-4 text-base text-gray-600 leading-relaxed">
              {DR_STAGE_DESCRIPTIONS[analysisResult.prediction] ||
                "No description available for this stage."}
            </p>
            <p
              className={`mt-2 text-lg ${getConfidenceColor(
                analysisResult.confidence
              )}`}
            >
              Confidence: {analysisResult.confidence.toFixed(1)}%
            </p>
            <p className="text-sm mt-2 text-gray-500">
              {analysisResult.additionalInfo}
            </p>
          </div>
        </div>
      ) : (
        <div className="gradient-card rounded-xl p-8 medical-shadow medical-border text-center">
          <div className="space-y-4">
            <div className="w-16 h-16 mx-auto rounded-full bg-gradient-to-r from-healthcare-sky to-healthcare-lavender flex items-center justify-center">
              <Brain className="h-8 w-8 text-white" />
            </div>
            <h3 className="text-xl font-semibold text-gray-700">
              Ready for Analysis
            </h3>
            <p className="text-gray-500">
              Upload a retinal fundus image to begin {moduleName.toLowerCase()}{" "}
              analysis.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default DRModuleAnalysis;
