
import React from 'react';
import { Eye, Brain, Clock, Search, Stethoscope } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface Module {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  color: string;
}

const modules: Module[] = [
  {
    id: 'dr_classification',
    name: 'DR Classification',
    description: 'Diabetic Retinopathy Detection & Classification',
    icon: <Eye className="h-5 w-5" />,
    color: 'from-red-400 to-red-600'
  },
  {
    id: 'vessel_segmentation',
    name: 'Vessel Segmentation',
    description: 'Retinal Blood Vessel Segmentation Analysis',
    icon: <Brain className="h-5 w-5" />,
    color: 'from-blue-400 to-blue-600'
  },
  {
    id: 'age_prediction',
    name: 'Age Prediction',
    description: 'Biological Age Estimation from Retinal Features',
    icon: <Clock className="h-5 w-5" />,
    color: 'from-green-400 to-green-600'
  },
  {
    id: 'myopia_detection',
    name: 'Myopia Detection',
    description: 'Myopia Risk Assessment & Detection',
    icon: <Search className="h-5 w-5" />,
    color: 'from-purple-400 to-purple-600'
  },
  {
    id: 'glaucoma_detection',
    name: 'Glaucoma Detection',
    description: 'Glaucoma Detection & Optic Cup Segmentation',
    icon: <Stethoscope className="h-5 w-5" />,
    color: 'from-orange-400 to-orange-600'
  }
];

interface ModuleSelectorProps {
  selectedModule: string;
  onModuleSelect: (moduleId: string) => void;
  disabled: boolean;
}

const ModuleSelector: React.FC<ModuleSelectorProps> = ({
  selectedModule,
  onModuleSelect,
  disabled
}) => {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-700 text-center">
        Select Analysis Module
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {modules.map((module) => (
          <Button
            key={module.id}
            variant={selectedModule === module.id ? "default" : "outline"}
            className={`h-auto p-4 flex flex-col items-start space-y-2 text-left transition-all duration-300 hover:scale-105 ${
              selectedModule === module.id
                ? 'gradient-button text-white shadow-lg'
                : 'bg-white/80 hover:bg-white medical-shadow medical-border'
            }`}
            onClick={() => onModuleSelect(module.id)}
            disabled={disabled}
          >
            <div className="flex items-center space-x-2 w-full">
              <div className={`p-2 rounded-lg bg-gradient-to-r ${module.color} text-white`}>
                {module.icon}
              </div>
              <div className="flex-1">
                <h4 className="font-medium text-sm leading-tight">
                  {module.name}
                </h4>
              </div>
            </div>
            <p className="text-xs opacity-80 leading-tight">
              {module.description}
            </p>
          </Button>
        ))}
      </div>
    </div>
  );
};

export default ModuleSelector;
