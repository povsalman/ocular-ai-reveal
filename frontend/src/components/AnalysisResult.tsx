
import React from 'react';
import { CheckCircle, AlertTriangle, Info, TrendingUp } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { AnalysisResult as AnalysisResultType } from '@/types/analysis';
import ReactMarkdown from 'react-markdown';

interface AnalysisResultProps {
  result: AnalysisResultType;
}

const AnalysisResult: React.FC<AnalysisResultProps> = ({ result }) => {
  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-green-600 bg-green-50 border-green-200';
      case 'medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'high': return 'text-red-600 bg-red-50 border-red-200';
      default: return 'text-blue-600 bg-blue-50 border-blue-200';
    }
  };

  const getRiskIcon = (level: string) => {
    switch (level) {
      case 'low': return <CheckCircle className="h-5 w-5" />;
      case 'medium': return <Info className="h-5 w-5" />;
      case 'high': return <AlertTriangle className="h-5 w-5" />;
      default: return <TrendingUp className="h-5 w-5" />;
    }
  };

  return (
    <div className="animate-fade-in space-y-6">
      <Card className="gradient-result medical-shadow medical-border">
        <CardContent className="p-6">
          <div className="space-y-6">
            {/* Header */}
            <div className="text-center">
              <h3 className="text-2xl font-bold text-gray-800 mb-2">
                Analysis Results
              </h3>
              <p className="text-sm text-gray-600 font-medium">
                {result.moduleName}
              </p>
            </div>

            {/* Main Prediction */}
            <div className="text-center space-y-4">
              <div className={`inline-flex items-center space-x-3 px-6 py-4 rounded-xl border-2 ${getRiskColor(result.riskLevel)}`}>
                {getRiskIcon(result.riskLevel)}
                <div className="text-left">
                  <div className="text-lg font-bold">
                    {result.prediction}
                  </div>
                  {result.details && (
                    <div className="text-sm opacity-80">
                      {result.details}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-4 bg-white/50 rounded-lg border border-white/30">
                <div className="text-2xl font-bold text-primary">
                  {result.accuracy.toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600 font-medium">
                  Model Accuracy
                </div>
              </div>
              
              <div className="text-center p-4 bg-white/50 rounded-lg border border-white/30">
                <div className="text-2xl font-bold text-primary">
                  {result.confidence.toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600 font-medium">
                  Confidence Level
                </div>
              </div>
            </div>

            {/* Additional Information */}
            {result.additionalInfo && (
              <div className="p-4 bg-white/40 rounded-lg border border-white/30">
                <div className="flex items-start space-x-3">
                  <Info className="h-5 w-5 text-blue-500 mt-0.5 flex-shrink-0" />
                  <div>
                    <h4 className="font-medium text-gray-800 mb-1">
                      Clinical Notes
                    </h4>
                    <div className="text-sm text-gray-600 leading-relaxed">
                      <ReactMarkdown>{result.additionalInfo}</ReactMarkdown>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Disclaimer */}
            <div className="text-center pt-4 border-t border-white/30">
              <p className="text-xs text-gray-500 leading-relaxed">
                <strong>Medical Disclaimer:</strong> This AI analysis is for research purposes only. 
                Please consult with a qualified healthcare professional for medical diagnosis and treatment.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Key Features Display */}
      {(result.features && Object.keys(result.features).length > 0) && (
        <Card className="gradient-result medical-shadow medical-border">
          <CardContent className="p-6">
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-800 text-center">
                Key features
              </h4>
              <div className="p-4 rounded-lg border border-green-200 bg-green-50 mb-4">
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead>
                      <tr>
                        <th className="text-left pr-4 pb-1 text-green-800">Feature</th>
                        <th className="text-left pb-1 text-green-800">Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(result.features).map(([key, value]) => (
                        <tr key={key}>
                          <td className="pr-4 py-1 text-green-900 font-medium">{key}</td>
                          <td className="py-1 text-green-900">{typeof value === 'number' ? value.toFixed(4) : value}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default AnalysisResult;
