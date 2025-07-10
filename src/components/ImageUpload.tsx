
import React, { useCallback, useState } from 'react';
import { Upload, Image as ImageIcon, X } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ImageUploadProps {
  onImageSelect: (file: File, preview: string) => void;
  selectedImage?: string;
  onRemoveImage: () => void;
  isAnalyzing: boolean;
}

const ImageUpload: React.FC<ImageUploadProps> = ({
  onImageSelect,
  selectedImage,
  onRemoveImage,
  isAnalyzing
}) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    const imageFile = files.find(file => file.type.startsWith('image/'));
    
    if (imageFile) {
      const preview = URL.createObjectURL(imageFile);
      onImageSelect(imageFile, preview);
    }
  }, [onImageSelect]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      const preview = URL.createObjectURL(file);
      onImageSelect(file, preview);
    }
  }, [onImageSelect]);

  if (selectedImage) {
    return (
      <div className="relative group">
        <div className="relative overflow-hidden rounded-xl medical-shadow medical-border">
          <img
            src={selectedImage}
            alt="Selected retinal fundus"
            className="w-full h-80 object-cover"
          />
          {isAnalyzing && (
            <div className="absolute inset-0 bg-black/20 flex items-center justify-center">
              <div className="bg-white/90 rounded-lg px-4 py-2 flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-primary border-t-transparent"></div>
                <span className="text-sm font-medium">Analyzing...</span>
              </div>
            </div>
          )}
        </div>
        <Button
          onClick={onRemoveImage}
          variant="destructive"
          size="sm"
          className="absolute -top-2 -right-2 rounded-full w-8 h-8 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
          disabled={isAnalyzing}
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    );
  }

  return (
    <div
      className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 cursor-pointer medical-shadow ${
        isDragOver
          ? 'border-primary bg-primary/5 scale-105'
          : 'border-gray-300 hover:border-primary/50 hover:bg-primary/5'
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => document.getElementById('file-input')?.click()}
    >
      <input
        id="file-input"
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        className="hidden"
      />
      
      <div className="flex flex-col items-center space-y-4">
        <div className="relative">
          <div className="w-16 h-16 rounded-full gradient-button flex items-center justify-center">
            <Upload className="h-8 w-8 text-white" />
          </div>
          <div className="absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-healthcare-lavender flex items-center justify-center">
            <ImageIcon className="h-3 w-3 text-white" />
          </div>
        </div>
        
        <div className="space-y-2">
          <h3 className="text-lg font-semibold text-gray-700">
            Upload Retinal Fundus Image
          </h3>
          <p className="text-sm text-gray-500 max-w-sm">
            Drag and drop your retinal image here, or click to browse
          </p>
          <p className="text-xs text-gray-400">
            Supports: PNG, JPG, JPEG formats
          </p>
        </div>
      </div>
    </div>
  );
};

export default ImageUpload;
