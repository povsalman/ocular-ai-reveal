
export interface AnalysisResult {
  moduleId: string;
  moduleName: string;
  prediction: string;
  accuracy: number;
  confidence: number;
  details?: string;
  riskLevel: 'low' | 'medium' | 'high';
  additionalInfo?: string;
  maskImage?: string; // For segmentation results
}
