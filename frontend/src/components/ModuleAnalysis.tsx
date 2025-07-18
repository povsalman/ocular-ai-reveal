import React, { useState } from 'react';
import { Brain, AlertTriangle, CheckCircle, XCircle, BarChart3 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import ImageUpload from '@/components/ImageUpload';
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
  cdr?: number;
  predicted_age?: number;
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
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);
  const { toast } = useToast();

  const handleImageSelect = (file: File, preview: string) => {
    setSelectedFile(file);
    setSelectedImage(preview);
    setAnalysisResult(null);
    const img = new window.Image();
    img.onload = () => {
      setImageDimensions({ width: img.width, height: img.height });
    };
    img.src = preview;
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
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('model_type', mapModelType(moduleId));

      const response = await fetch('http://localhost:8000/predict/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const apiResult: APIResponse = await response.json();

      if (apiResult.status !== 'success') throw new Error(`Prediction failed: ${apiResult.status}`);

      const result: AnalysisResultType = {
        moduleId: moduleId,
        moduleName: moduleName,
        prediction: apiResult.predicted_class,
        accuracy: apiResult.metrics?.accuracy || 0,
        confidence: apiResult.confidence * 100,
        details: `Dataset: ${apiResult.dataset_used || 'Unknown'}`,
        riskLevel: mapRiskLevel(apiResult.confidence * 100),
        maskImage: (moduleId !== 'age_prediction' && apiResult.mask_image) ? `data:image/png;base64,${apiResult.mask_image}` : undefined,
        additionalInfo: moduleId === 'age_prediction'
          ? undefined
          : `Our vessel segmentation model uses the R2UNet architecture, specifically designed for biomedical image segmentation. This model provides precise pixel-level segmentation of retinal blood vessels, enabling detailed vascular analysis and pathology detection.`,
        metrics: apiResult.metrics,
        cdr: apiResult.cdr,
        numericAge: moduleId === 'age_prediction' ? apiResult.predicted_age : undefined,
        modelName: moduleId === 'age_prediction' ? 'InceptionResnetV2' : undefined
      };

      setAnalysisResult(result);

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

      {/* Results Section would go here. You can plug in the cleaned JSX section if needed */}
    </div>
  );
};

export default ModuleAnalysis;
