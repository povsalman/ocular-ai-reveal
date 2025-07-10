
import React, { useState } from 'react';
import { Brain, Sparkles, Activity } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import ImageUpload from '@/components/ImageUpload';
import ModuleSelector from '@/components/ModuleSelector';
import AnalysisResult from '@/components/AnalysisResult';

interface AnalysisResult {
  moduleId: string;
  moduleName: string;
  prediction: string;
  accuracy: number;
  confidence: number;
  details?: string;
  riskLevel: 'low' | 'medium' | 'high';
  additionalInfo?: string;
}

const Index = () => {
  const [selectedImage, setSelectedImage] = useState<string>('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedModule, setSelectedModule] = useState<string>('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
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

  const generateMockResult = (moduleId: string): AnalysisResult => {
    const moduleData = {
      dr_classification: {
        name: 'DR Classification',
        predictions: [
          { prediction: 'No Diabetic Retinopathy', risk: 'low' as const, details: 'Healthy retinal structure detected' },
          { prediction: 'Mild Diabetic Retinopathy', risk: 'medium' as const, details: 'Early stage changes observed' },
          { prediction: 'Moderate Diabetic Retinopathy', risk: 'high' as const, details: 'Significant retinal changes detected' }
        ]
      },
      vessel_segmentation: {
        name: 'Vessel Segmentation',
        predictions: [
          { prediction: 'Normal Vessel Architecture', risk: 'low' as const, details: 'Healthy vascular pattern' },
          { prediction: 'Mild Vessel Irregularities', risk: 'medium' as const, details: 'Minor vascular changes noted' },
          { prediction: 'Significant Vessel Changes', risk: 'high' as const, details: 'Notable vascular abnormalities' }
        ]
      },
      age_prediction: {
        name: 'Age Prediction',
        predictions: [
          { prediction: 'Estimated Age: 45-50 years', risk: 'low' as const, details: 'Age-appropriate retinal features' },
          { prediction: 'Estimated Age: 55-60 years', risk: 'medium' as const, details: 'Moderate age-related changes' },
          { prediction: 'Estimated Age: 65+ years', risk: 'medium' as const, details: 'Advanced age-related features' }
        ]
      },
      myopia_detection: {
        name: 'Myopia Detection',
        predictions: [
          { prediction: 'No Myopia Detected', risk: 'low' as const, details: 'Normal refractive indicators' },
          { prediction: 'Mild Myopia Risk', risk: 'medium' as const, details: 'Early myopic changes observed' },
          { prediction: 'High Myopia Risk', risk: 'high' as const, details: 'Significant myopic features detected' }
        ]
      },
      glaucoma_detection: {
        name: 'Glaucoma Detection',
        predictions: [
          { prediction: 'No Glaucoma Detected', risk: 'low' as const, details: 'Healthy optic nerve structure' },
          { prediction: 'Glaucoma Suspect', risk: 'medium' as const, details: 'Suspicious optic nerve changes' },
          { prediction: 'Glaucoma Detected', risk: 'high' as const, details: 'Glaucomatous optic nerve damage' }
        ]
      }
    };

    const moduleInfo = moduleData[moduleId as keyof typeof moduleData];
    const randomPrediction = moduleInfo.predictions[Math.floor(Math.random() * moduleInfo.predictions.length)];
    
    return {
      moduleId,
      moduleName: moduleInfo.name,
      prediction: randomPrediction.prediction,
      accuracy: 89.5 + Math.random() * 8,
      confidence: 82.3 + Math.random() * 15,
      details: randomPrediction.details,
      riskLevel: randomPrediction.risk,
      additionalInfo: `Based on advanced deep learning analysis of retinal fundus images. This assessment uses state-of-the-art computer vision algorithms trained on extensive medical datasets.`
    };
  };

  const handleAnalyze = async () => {
    if (!selectedFile || !selectedModule) {
      toast({
        title: "Missing Information",
        description: "Please select both an image and analysis module.",
        variant: "destructive"
      });
      return;
    }

    setIsAnalyzing(true);
    setAnalysisResult(null);

    // Simulate AI processing time
    await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 2000));

    const result = generateMockResult(selectedModule);
    setAnalysisResult(result);
    setIsAnalyzing(false);

    toast({
      title: "Analysis Complete",
      description: `${result.moduleName} analysis finished successfully.`
    });
  };

  return (
    <div className="min-h-screen p-4 md:p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center space-x-3">
            <div className="relative">
              <Brain className="h-10 w-10 text-primary" />
              <Sparkles className="h-4 w-4 text-healthcare-lavender absolute -top-1 -right-1" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-primary to-healthcare-lavender bg-clip-text text-transparent">
              AI Retinal Analytics
            </h1>
          </div>
          
          <p className="text-lg text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Advanced artificial intelligence for comprehensive retinal fundus image analysis. 
            Upload your retinal image and select from our suite of specialized diagnostic modules.
          </p>
          
          <div className="flex items-center justify-center space-x-6 text-sm text-gray-500">
            <div className="flex items-center space-x-2">
              <Activity className="h-4 w-4 text-green-500" />
              <span>5 AI Modules</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse-soft"></div>
              <span>Real-time Analysis</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
              <span>Medical Grade</span>
            </div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Column - Upload & Modules */}
          <div className="space-y-6">
            <div className="gradient-card rounded-xl p-6 medical-shadow medical-border">
              <ImageUpload
                onImageSelect={handleImageSelect}
                selectedImage={selectedImage}
                onRemoveImage={handleRemoveImage}
                isAnalyzing={isAnalyzing}
              />
            </div>

            <div className="gradient-card rounded-xl p-6 medical-shadow medical-border">
              <ModuleSelector
                selectedModule={selectedModule}
                onModuleSelect={setSelectedModule}
                disabled={isAnalyzing}
              />
            </div>

            <div className="text-center">
              <Button
                onClick={handleAnalyze}
                disabled={!selectedFile || !selectedModule || isAnalyzing}
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
          </div>

          {/* Right Column - Results */}
          <div className="space-y-6">
            {analysisResult ? (
              <AnalysisResult result={analysisResult} />
            ) : (
              <div className="gradient-card rounded-xl p-8 medical-shadow medical-border text-center">
                <div className="space-y-4">
                  <div className="w-16 h-16 mx-auto rounded-full bg-gradient-to-r from-healthcare-sky to-healthcare-lavender flex items-center justify-center">
                    <Activity className="h-8 w-8 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-700">
                    Ready for Analysis
                  </h3>
                  <p className="text-gray-500">
                    Upload a retinal fundus image and select an analysis module to begin. 
                    Our AI will provide detailed insights with accuracy metrics.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="text-center py-8 border-t border-white/30">
          <p className="text-sm text-gray-500">
            Powered by advanced machine learning • For research and educational purposes • 
            Always consult healthcare professionals for medical decisions
          </p>
        </div>
      </div>
    </div>
  );
};

export default Index;
