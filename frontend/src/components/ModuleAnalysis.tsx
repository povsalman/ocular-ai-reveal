
import React, { useState } from 'react';
import { Brain, AlertTriangle, CheckCircle, XCircle, BarChart3 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import ImageUpload from '@/components/ImageUpload';
import AnalysisResult from '@/components/AnalysisResult';
import { AnalysisResult as AnalysisResultType } from '@/types/analysis';

interface ModuleAnalysisProps {
  moduleId: string;
  moduleName: string;
  generateMockResult: (moduleId: string) => AnalysisResultType;
}

interface APIMetrics {
  dice_coefficient: number;
  sensitivity: number;
  specificity: number;
  f1_score: number;
  accuracy: number;
  jaccard_similarity: number;
  auc: number;
}

interface APIResponse {
  status: string;
  predicted_class: string;
  confidence: number;
  model_type: string;
  dataset_used?: string;
  metrics?: APIMetrics;
  mask_image?: string;
  features?: Record<string, number | string>;
}

const ModuleAnalysis: React.FC<ModuleAnalysisProps> = ({
  moduleId,
  moduleName,
  generateMockResult
}) => {
  const [selectedImage, setSelectedImage] = useState<string>('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResultType | null>(null);
  const { toast } = useToast();

  const handleImageSelect = (file: File, preview: string) => {
    setSelectedFile(file);
    setSelectedImage(preview);
    setAnalysisResult(null);
    console.log('Image selected:', file.name, file.type);
  };

  const handleRemoveImage = () => {
    setSelectedImage('');
    setSelectedFile(null);
    setAnalysisResult(null);
  };

  const mapModelType = (moduleId: string): string => {
    const modelMap: { [key: string]: string } = {
      'vessel_segmentation': 'vessel',
      'dr_classification': 'dr',
      'glaucoma_detection': 'glaucoma',
      'age_prediction': 'age',
      'myopia_detection': 'myopia'
    };
    return modelMap[moduleId] || 'vessel';
  };

  const mapRiskLevel = (confidence: number): 'low' | 'medium' | 'high' => {
    if (confidence >= 80) return 'low';
    if (confidence >= 60) return 'medium';
    return 'high';
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 80) return 'text-green-600';
    if (confidence >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceIcon = (confidence: number) => {
    if (confidence >= 80) return <CheckCircle className="h-5 w-5 text-green-600" />;
    if (confidence >= 60) return <AlertTriangle className="h-5 w-5 text-yellow-600" />;
    return <XCircle className="h-5 w-5 text-red-600" />;
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      toast({
        title: "Missing Image",
        description: "Please select an image to analyze.",
        variant: "destructive"
      });
      return;
    }

    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      // Create FormData for the API request
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('model_type', mapModelType(moduleId));

      // Make API call to backend
      const response = await fetch('http://localhost:8000/predict/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const apiResult: APIResponse = await response.json();
      
      // Check if prediction was successful
      if (apiResult.status !== 'success') {
        throw new Error(`Prediction failed: ${apiResult.status}`);
      }

      // Convert API response to AnalysisResult format
      const result: AnalysisResultType = {
        moduleId: moduleId,
        moduleName: moduleName,
        prediction: apiResult.predicted_class,
        accuracy: apiResult.metrics?.accuracy || 0,
        confidence: apiResult.confidence * 100, // Convert to percentage
        details: `Dataset: ${apiResult.dataset_used || 'Unknown'}`,
        riskLevel: mapRiskLevel(apiResult.confidence * 100),
        maskImage: apiResult.mask_image ? `data:image/png;base64,${apiResult.mask_image}` : undefined,
        additionalInfo: `Our vessel segmentation model uses the R2UNet architecture, specifically designed for biomedical image segmentation. This model provides precise pixel-level segmentation of retinal blood vessels, enabling detailed vascular analysis and pathology detection.`,
        // Add metrics for display
        metrics: apiResult.metrics,
        features: apiResult.features || undefined,
      };

      setAnalysisResult(result);

      // Show warning for low confidence
      if (apiResult.metrics?.dice_coefficient && apiResult.metrics.dice_coefficient < 0.4) {
        toast({
          title: "Low Confidence Warning",
          description: "The model's prediction confidence is low. Please consider consulting a medical expert.",
          variant: "destructive"
        });
      } else {
        toast({
          title: "Analysis Complete",
          description: `${moduleName} analysis finished successfully.`
        });
      }

    } catch (error) {
      console.error('Analysis error:', error);
      
      // Fallback to mock result if API fails
      const mockResult = generateMockResult(moduleId);
      setAnalysisResult(mockResult);
      
      toast({
        title: "Analysis Complete (Mock)",
        description: `${moduleName} analysis completed using mock data due to API error.`
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload Section */}
      <div className="gradient-card rounded-xl p-6 medical-shadow medical-border">
        <ImageUpload
          onImageSelect={handleImageSelect}
          selectedImage={selectedImage}
          onRemoveImage={handleRemoveImage}
          isAnalyzing={isAnalyzing}
        />
      </div>

      {/* Analysis Button */}
      <div className="text-center">
        <Button
          onClick={handleAnalyze}
          disabled={!selectedFile || isAnalyzing}
          className="gradient-button text-white px-8 py-3 text-lg font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
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

      {/* Results Section */}
      {analysisResult ? (
        <div className="space-y-6">
          {/* Two-column layout for image and mask */}
          <div className="grid lg:grid-cols-2 gap-6">
            {/* Original Image */}
            <div className="gradient-card rounded-xl p-6 medical-shadow medical-border">
              <h3 className="text-lg font-semibold text-gray-700 mb-4">Original Image</h3>
              {selectedImage && (
                <div className="flex justify-center">
                  <img 
                    src={selectedImage} 
                    alt="Uploaded retinal image" 
                    className="max-w-full max-h-96 object-contain rounded-lg"
                    style={{ 
                      aspectRatio: 'auto',
                      width: 'auto',
                      height: 'auto'
                    }}
                  />
                </div>
              )}
            </div>

            {/* Segmentation Mask or Key Features */}
            {moduleId === 'myopia_detection' ? (
              <div className="gradient-card rounded-xl p-6 medical-shadow medical-border">
                <h3 className="text-lg font-semibold text-gray-700 mb-4">Key features</h3>
                {analysisResult.features && Object.keys(analysisResult.features).length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="min-w-full text-sm">
                      <thead>
                        <tr>
                          <th className="text-left pr-4 pb-1 text-green-800">Feature</th>
                          <th className="text-left pb-1 text-green-800">Value</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(analysisResult.features)
                          .filter(([key]) => key !== 'min_vessel_width' && key !== 'branch_density')
                          .map(([key, value]) => (
                            <tr key={key}>
                              <td className="pr-4 py-1 text-green-900 font-medium">{key}</td>
                              <td className="py-1 text-green-900">{typeof value === 'number' ? value.toFixed(4) : value}</td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="text-gray-500">No features extracted.</div>
                )}
              </div>
            ) : (
              <div className="gradient-card rounded-xl p-6 medical-shadow medical-border">
                <h3 className="text-lg font-semibold text-gray-700 mb-4">Segmentation Mask</h3>
                {analysisResult.maskImage ? (
                  <div className="flex justify-center">
                    <img 
                      src={analysisResult.maskImage} 
                      alt="Vessel segmentation mask" 
                      className="max-w-full max-h-96 object-contain rounded-lg"
                      style={{ 
                        aspectRatio: 'auto',
                        width: 'auto',
                        height: 'auto'
                      }}
                    />
                  </div>
                ) : (
                  <div className="w-full h-64 bg-gray-100 rounded-lg flex items-center justify-center">
                    <XCircle className="h-12 w-12 text-gray-400" />
                    <span className="ml-2 text-gray-500">No mask available</span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Metrics Section */}
          <div className="gradient-card rounded-xl p-6 medical-shadow medical-border">
            <div className="flex items-center space-x-2 mb-4">
              <BarChart3 className="h-5 w-5 text-gray-600" />
              <h3 className="text-lg font-semibold text-gray-700">Analysis Results</h3>
            </div>

            {/* Prediction and Confidence */}
            <div className="grid md:grid-cols-2 gap-4 mb-6">
              <div className="space-y-2">
                <p className="text-sm text-gray-600">Prediction</p>
                <p className="text-lg font-semibold text-gray-800">{analysisResult.prediction}</p>
              </div>
              <div className="space-y-2">
                <p className="text-sm text-gray-600">Confidence</p>
                <div className="flex items-center space-x-2">
                  {getConfidenceIcon(analysisResult.confidence)}
                  <span className={`text-lg font-semibold ${getConfidenceColor(analysisResult.confidence)}`}>
                    {analysisResult.confidence.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Metrics Grid */}
            {analysisResult.metrics && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-3 bg-blue-50 rounded-lg">
                  <p className="text-xs text-gray-600 mb-1">Dice Coefficient</p>
                  <p className="text-lg font-bold text-blue-600">
                    {(analysisResult.metrics.dice_coefficient * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-center p-3 bg-green-50 rounded-lg">
                  <p className="text-xs text-gray-600 mb-1">Sensitivity</p>
                  <p className="text-lg font-bold text-green-600">
                    {(analysisResult.metrics.sensitivity * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-center p-3 bg-purple-50 rounded-lg">
                  <p className="text-xs text-gray-600 mb-1">Specificity</p>
                  <p className="text-lg font-bold text-purple-600">
                    {(analysisResult.metrics.specificity * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-center p-3 bg-orange-50 rounded-lg">
                  <p className="text-xs text-gray-600 mb-1">F1 Score</p>
                  <p className="text-lg font-bold text-orange-600">
                    {(analysisResult.metrics.f1_score * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-center p-3 bg-red-50 rounded-lg">
                  <p className="text-xs text-gray-600 mb-1">Accuracy</p>
                  <p className="text-lg font-bold text-red-600">
                    {(analysisResult.metrics.accuracy * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-center p-3 bg-indigo-50 rounded-lg">
                  <p className="text-xs text-gray-600 mb-1">Jaccard Similarity</p>
                  <p className="text-lg font-bold text-indigo-600">
                    {(analysisResult.metrics.jaccard_similarity * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-center p-3 bg-teal-50 rounded-lg">
                  <p className="text-xs text-gray-600 mb-1">AUC</p>
                  <p className="text-lg font-bold text-teal-600">
                    {(analysisResult.metrics.auc * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <p className="text-xs text-gray-600 mb-1">Dataset Used</p>
                  <p className="text-lg font-bold text-gray-600">
                    {analysisResult.details.split(': ')[1] || 'Unknown'}
                  </p>
                </div>
              </div>
            )}

            {/* Warning for low confidence */}
            {analysisResult.metrics?.dice_coefficient && analysisResult.metrics.dice_coefficient < 0.4 && (
              <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                <div className="flex items-center space-x-2">
                  <AlertTriangle className="h-5 w-5 text-yellow-600" />
                  <span className="text-yellow-800 font-medium">
                    ⚠️ The model's prediction confidence is low. Please consider consulting a medical expert.
                  </span>
                </div>
              </div>
            )}
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
              Upload a retinal fundus image to begin {moduleName.toLowerCase()} analysis. 
              Our AI will provide detailed insights with accuracy metrics.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModuleAnalysis;
