
import React, { useState } from 'react';
import { Brain } from 'lucide-react';
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

    // Simulate AI processing time
    await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 2000));

    const result = generateMockResult(moduleId);
    setAnalysisResult(result);
    setIsAnalyzing(false);

    toast({
      title: "Analysis Complete",
      description: `${moduleName} analysis finished successfully.`
    });
  };

  return (
    <div className="grid lg:grid-cols-2 gap-8">
      {/* Left Column - Upload & Analysis */}
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
      </div>

      {/* Right Column - Results */}
      <div className="space-y-6">
        {analysisResult ? (
          <AnalysisResult result={analysisResult} />
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
    </div>
  );
};

export default ModuleAnalysis;
