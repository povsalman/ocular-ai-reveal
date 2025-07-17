
import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Brain, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ModuleLayoutProps {
  children: React.ReactNode;
  title: string;
  description: string;
  icon: React.ReactNode;
}

const ModuleLayout: React.FC<ModuleLayoutProps> = ({ children, title, description, icon }) => {
  return (
    <div className="min-h-screen p-4 md:p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <Link to="/">
            <Button variant="outline" className="flex items-center space-x-2">
              <ArrowLeft className="h-4 w-4" />
              <span>Back to Home</span>
            </Button>
          </Link>
          
          <div className="flex items-center space-x-3">
            <Brain className="h-8 w-8 text-primary" />
            <Sparkles className="h-5 w-5 text-healthcare-lavender" />
          </div>
        </div>

        {/* Module Header */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center space-x-4">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-r from-primary to-healthcare-lavender text-white flex items-center justify-center">
              {icon}
            </div>
            <div className="text-left">
              <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-primary to-healthcare-lavender bg-clip-text text-transparent">
                {title}
              </h1>
              <p className="text-lg text-gray-600 mt-2">
                {description}
              </p>
            </div>
          </div>
        </div>

        {/* Content */}
        {children}
      </div>
    </div>
  );
};

export default ModuleLayout;
