
import React from 'react';
import { Brain, Sparkles, Activity, Eye, Clock, Search, Stethoscope } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Link } from 'react-router-dom';

const Index = () => {
  const modules = [
    {
      id: 'dr-classification',
      name: 'DR Classification',
      description: 'Diabetic Retinopathy Detection & Classification',
      icon: <Eye className="h-8 w-8" />,
      color: 'from-red-400 to-red-600',
      path: '/dr-classification'
    },
    {
      id: 'vessel-segmentation',
      name: 'Vessel Segmentation',
      description: 'Retinal Blood Vessel Segmentation using R2UNet',
      icon: <Brain className="h-8 w-8" />,
      color: 'from-blue-400 to-blue-600',
      path: '/vessel-segmentation'
    },
    {
      id: 'age-prediction',
      name: 'Age Prediction',
      description: 'Biological Age Estimation from Retinal Features',
      icon: <Clock className="h-8 w-8" />,
      color: 'from-green-400 to-green-600',
      path: '/age-prediction'
    },
    {
      id: 'myopia-detection',
      name: 'Myopia Detection',
      description: 'Myopia Risk Assessment & Detection',
      icon: <Search className="h-8 w-8" />,
      color: 'from-purple-400 to-purple-600',
      path: '/myopia-detection'
    },
    {
      id: 'glaucoma-detection',
      name: 'Glaucoma Detection',
      description: 'Glaucoma Detection & Optic Cup Segmentation',
      icon: <Stethoscope className="h-8 w-8" />,
      color: 'from-orange-400 to-orange-600',
      path: '/glaucoma-detection'
    }
  ];

  return (
    <div className="min-h-screen p-4 md:p-8">
      <div className="max-w-6xl mx-auto space-y-12">
        {/* Header */}
        <div className="text-center space-y-6">
          <div className="flex items-center justify-center space-x-3">
            <div className="relative">
              <Brain className="h-12 w-12 text-primary" />
              <Sparkles className="h-5 w-5 text-healthcare-lavender absolute -top-1 -right-1" />
            </div>
            <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-primary to-healthcare-lavender bg-clip-text text-transparent">
              AI Retinal Analytics
            </h1>
          </div>
          
          <p className="text-xl text-gray-600 max-w-4xl mx-auto leading-relaxed">
            Advanced artificial intelligence for comprehensive retinal fundus image analysis. 
            Explore our suite of specialized diagnostic modules powered by state-of-the-art deep learning algorithms.
          </p>
          
          <div className="flex items-center justify-center space-x-8 text-sm text-gray-500">
            <div className="flex items-center space-x-2">
              <Activity className="h-5 w-5 text-green-500" />
              <span>5 AI Modules</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse-soft"></div>
              <span>Real-time Analysis</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              <span>Medical Grade</span>
            </div>
          </div>
        </div>

        {/* Modules Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {modules.map((module) => (
            <Link
              key={module.id}
              to={module.path}
              className="group gradient-card rounded-2xl p-8 medical-shadow medical-border hover:scale-105 transition-all duration-300 hover:shadow-2xl"
            >
              <div className="space-y-6">
                <div className={`w-16 h-16 rounded-2xl bg-gradient-to-r ${module.color} text-white flex items-center justify-center group-hover:scale-110 transition-transform duration-300`}>
                  {module.icon}
                </div>
                
                <div className="space-y-3">
                  <h3 className="text-xl font-bold text-gray-800 group-hover:text-primary transition-colors">
                    {module.name}
                  </h3>
                  <p className="text-gray-600 leading-relaxed">
                    {module.description}
                  </p>
                </div>

                <Button 
                  className="w-full gradient-button text-white font-semibold py-3 rounded-xl group-hover:shadow-lg transition-all duration-300"
                >
                  Explore Module
                </Button>
              </div>
            </Link>
          ))}
        </div>

        {/* Features Section */}
        <div className="gradient-card rounded-2xl p-8 md:p-12 medical-shadow medical-border text-center">
          <div className="space-y-6">
            <h2 className="text-3xl font-bold text-gray-800">
              Why Choose AI Retinal Analytics?
            </h2>
            
            <div className="grid md:grid-cols-3 gap-8">
              <div className="space-y-3">
                <div className="w-12 h-12 bg-gradient-to-r from-healthcare-sky to-healthcare-lavender rounded-xl flex items-center justify-center mx-auto">
                  <Brain className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-lg font-semibold text-gray-700">Advanced AI</h3>
                <p className="text-gray-600 text-sm">
                  State-of-the-art deep learning models trained on extensive medical datasets
                </p>
              </div>
              
              <div className="space-y-3">
                <div className="w-12 h-12 bg-gradient-to-r from-healthcare-sky to-healthcare-lavender rounded-xl flex items-center justify-center mx-auto">
                  <Activity className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-lg font-semibold text-gray-700">Real-time Analysis</h3>
                <p className="text-gray-600 text-sm">
                  Fast and accurate analysis with instant results and detailed insights
                </p>
              </div>
              
              <div className="space-y-3">
                <div className="w-12 h-12 bg-gradient-to-r from-healthcare-sky to-healthcare-lavender rounded-xl flex items-center justify-center mx-auto">
                  <Sparkles className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-lg font-semibold text-gray-700">Research Grade</h3>
                <p className="text-gray-600 text-sm">
                  Professional-grade analysis tools for research and educational purposes
                </p>
              </div>
            </div>
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
