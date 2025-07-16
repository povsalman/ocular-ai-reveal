
export interface AnalysisResult {
  moduleId: string;
  moduleName: string;
  prediction: string;
  confidence: number;
  details?: string;
  additionalInfo?: string;
  maskImage?: string; // For segmentation results
  metrics?: {
    dice_coefficient: number;
    sensitivity: number;
    specificity: number;
    f1_score: number;
    accuracy: number;
    jaccard_similarity: number;
    auc: number;
  };
}
