'use client'

import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useDropzone } from 'react-dropzone'
import { 
  DocumentArrowUpIcon,
  XMarkIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  CogIcon
} from '@heroicons/react/24/outline'
import { useUploadDocuments } from '@/hooks/use-api'
import toast from 'react-hot-toast'

interface UploadSettings {
  extractContracts: boolean
  chunkSize: number
  enableOCR: boolean
  autoProcess: boolean
}

export default function DocumentUpload() {
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const [uploadSettings, setUploadSettings] = useState<UploadSettings>({
    extractContracts: true,
    chunkSize: 1000,
    enableOCR: true,
    autoProcess: true
  })
  const [isProcessing, setIsProcessing] = useState(false)
  const [uploadResults, setUploadResults] = useState<any[]>([])
  const [showSettings, setShowSettings] = useState(false)
  
  const uploadMutation = useUploadDocuments()

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setUploadedFiles(prev => [...prev, ...acceptedFiles])
    
    if (uploadSettings.autoProcess) {
      handleUpload([...uploadedFiles, ...acceptedFiles])
    }
  }, [uploadedFiles, uploadSettings.autoProcess])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'image/*': ['.jpg', '.jpeg', '.png']
    },
    multiple: true
  })

  const handleUpload = async (files: File[] = uploadedFiles) => {
    if (files.length === 0) return
    
    setIsProcessing(true)
    setUploadResults([])
    
    try {
      const results = await uploadMutation.mutateAsync(files)
      setUploadResults(results.results || [])
      
      const successCount = results.results?.filter((r: any) => r.success).length || 0
      const failCount = results.results?.filter((r: any) => !r.success).length || 0
      
      if (successCount > 0) {
        toast.success(`Successfully processed ${successCount} documents!`)
      }
      if (failCount > 0) {
        toast.error(`Failed to process ${failCount} documents`)
      }
      
    } catch (error) {
      toast.error('Upload failed')
      console.error('Upload error:', error)
    } finally {
      setIsProcessing(false)
    }
  }

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index))
  }

  const clearAll = () => {
    setUploadedFiles([])
    setUploadResults([])
  }

  return (
    <div className="h-full overflow-auto bg-gradient-to-br from-primary-50 via-white to-accent-50">
      <div className="max-w-6xl mx-auto p-6">
        <div className="flex justify-between items-center mb-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary-600 to-accent-500 mb-3">
              Document Upload & Processing
            </h1>
            <p className="text-secondary-600 text-lg">
              Upload contracts and documents for AI-powered analysis and insights
            </p>
          </motion.div>
          
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setShowSettings(!showSettings)}
            className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition-all duration-200"
          >
            <CogIcon className="h-5 w-5 text-primary-600" />
            <span className="font-medium">Settings</span>
          </motion.button>
        </div>

        {/* Upload Settings */}
        <AnimatePresence>
          {showSettings && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="bg-white rounded-xl shadow-lg p-6 mb-8 overflow-hidden border border-gray-100"
            >
              <h2 className="text-xl font-semibold text-secondary-900 mb-5 flex items-center">
                <span className="bg-primary-100 text-primary-600 p-2 rounded-lg mr-3">
                  <CogIcon className="h-5 w-5" />
                </span>
                Upload Settings
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="space-y-5">
                  <label className="flex items-start space-x-3 p-3 hover:bg-gray-50 rounded-lg transition-colors">
                    <input
                      type="checkbox"
                      checked={uploadSettings.extractContracts}
                      onChange={(e) => setUploadSettings(prev => ({ ...prev, extractContracts: e.target.checked }))}
                      className="mt-1 rounded border-secondary-300 text-primary-600 focus:ring-primary-500 h-5 w-5"
                    />
                    <div>
                      <span className="text-sm font-medium text-secondary-900">Extract Contract Details</span>
                      <p className="text-xs text-secondary-600 mt-1">Automatically identify and extract contract information</p>
                    </div>
                  </label>
                  
                  <label className="flex items-start space-x-3 p-3 hover:bg-gray-50 rounded-lg transition-colors">
                    <input
                      type="checkbox"
                      checked={uploadSettings.enableOCR}
                      onChange={(e) => setUploadSettings(prev => ({ ...prev, enableOCR: e.target.checked }))}
                      className="mt-1 rounded border-secondary-300 text-primary-600 focus:ring-primary-500 h-5 w-5"
                    />
                    <div>
                      <span className="text-sm font-medium text-secondary-900">Enable OCR</span>
                      <p className="text-xs text-secondary-600 mt-1">Extract text from scanned documents and images</p>
                    </div>
                  </label>
                </div>
                
                <div className="space-y-5">
                  <div className="p-3 hover:bg-gray-50 rounded-lg transition-colors">
                    <label className="block text-sm font-medium text-secondary-900 mb-2 flex justify-between">
                      <span>Chunk Size</span>
                      <span className="bg-primary-100 text-primary-600 px-2 py-1 rounded text-xs font-semibold">{uploadSettings.chunkSize}</span>
                    </label>
                    <input
                      type="range"
                      min="500"
                      max="2000"
                      step="100"
                      value={uploadSettings.chunkSize}
                      onChange={(e) => setUploadSettings(prev => ({ ...prev, chunkSize: parseInt(e.target.value) }))}
                      className="w-full accent-primary-600"
                    />
                    <p className="text-xs text-secondary-600 mt-1">Size of text chunks for processing (500-2000 characters)</p>
                  </div>
                  
                  <label className="flex items-start space-x-3 p-3 hover:bg-gray-50 rounded-lg transition-colors">
                    <input
                      type="checkbox"
                      checked={uploadSettings.autoProcess}
                      onChange={(e) => setUploadSettings(prev => ({ ...prev, autoProcess: e.target.checked }))}
                      className="mt-1 rounded border-secondary-300 text-primary-600 focus:ring-primary-500 h-5 w-5"
                    />
                    <div>
                      <span className="text-sm font-medium text-secondary-900">Auto-process on Upload</span>
                      <p className="text-xs text-secondary-600 mt-1">Start processing immediately after upload</p>
                    </div>
                  </label>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Upload Area */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="bg-white rounded-xl shadow-lg mb-8 overflow-hidden"
        >
          <div
            {...getRootProps()}
            className={`p-10 border-2 border-dashed rounded-lg transition-all duration-300 cursor-pointer ${
              isDragActive
                ? 'border-primary-400 bg-primary-50'
                : 'border-secondary-300 hover:border-primary-400 hover:bg-primary-50/50'
            }`}
          >
            <input {...getInputProps()} />
            <div className="text-center">
              <motion.div 
                initial={{ scale: 0.8 }}
                animate={{ scale: 1 }}
                transition={{ 
                  type: "spring", 
                  stiffness: 260, 
                  damping: 20 
                }}
              >
                <div className="w-20 h-20 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-5">
                  <DocumentArrowUpIcon className="w-10 h-10 text-primary-600" />
                </div>
              </motion.div>
              <h3 className="text-xl font-semibold text-secondary-900 mb-3">
                {isDragActive ? 'Drop files here...' : 'Upload Documents'}
              </h3>
              <p className="text-secondary-600 mb-4 text-lg">
                Drag & drop files here, or click to browse
              </p>
              <p className="text-sm text-secondary-500 bg-gray-50 inline-block px-4 py-2 rounded-full">
                Supported: PDF, DOCX, TXT, Images • Max 50MB per file
              </p>
            </div>
          </div>
        </motion.div>

        {/* File List */}
        {uploadedFiles.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="bg-white rounded-xl shadow-lg p-6 mb-8"
          >
            <div className="flex justify-between items-center mb-5">
              <h3 className="text-xl font-semibold text-secondary-900 flex items-center">
                <span className="bg-accent-100 text-accent-600 p-2 rounded-lg mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M3 5a2 2 0 012-2h10a2 2 0 012 2v10a2 2 0 01-2 2H5a2 2 0 01-2-2V5zm2 1v8h10V6H5z" clipRule="evenodd" />
                  </svg>
                </span>
                Files to Upload ({uploadedFiles.length})
              </h3>
              <div className="flex gap-3">
                {!uploadSettings.autoProcess && (
                  <motion.button
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                    onClick={() => handleUpload()}
                    disabled={isProcessing}
                    className="px-4 py-2 bg-gradient-to-r from-primary-600 to-primary-500 text-white rounded-lg shadow-sm hover:shadow-md transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  >
                    {isProcessing ? (
                      <>
                        <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span>Processing...</span>
                      </>
                    ) : (
                      <>
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                        </svg>
                        <span>Upload All</span>
                      </>
                    )}
                  </motion.button>
                )}
                <motion.button 
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={clearAll} 
                  className="px-4 py-2 border border-gray-200 text-secondary-700 rounded-lg hover:bg-gray-50 transition-all duration-200 flex items-center gap-2"
                >
                  <XMarkIcon className="h-4 w-4" />
                  <span>Clear All</span>
                </motion.button>
              </div>
            </div>
            
            <div className="space-y-3 max-h-64 overflow-y-auto pr-2 custom-scrollbar">
              {uploadedFiles.map((file, index) => (
                <motion.div 
                  key={index} 
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3 }}
                  className="flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-100 transition-colors"
                >
                  <div className="flex items-center gap-3 flex-1">
                    <div className="bg-white p-2 rounded-md shadow-sm">
                      <DocumentArrowUpIcon className="h-6 w-6 text-primary-500" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-secondary-900">{file.name}</p>
                      <p className="text-xs text-secondary-600">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                    </div>
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.1, rotate: 90 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => removeFile(index)}
                    className="p-2 hover:bg-red-100 rounded-full text-secondary-600 hover:text-red-600 transition-colors"
                  >
                    <XMarkIcon className="h-5 w-5" />
                  </motion.button>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Processing Status */}
        {isProcessing && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="bg-white rounded-xl shadow-lg p-6 mb-8 border-l-4 border-primary-500"
          >
            <div className="flex items-center gap-4 mb-4">
              <div className="relative">
                <div className="w-10 h-10 border-4 border-primary-100 rounded-full"></div>
                <div className="absolute top-0 left-0 w-10 h-10 border-4 border-primary-500 rounded-full border-t-transparent animate-spin"></div>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-secondary-900">Processing documents</h3>
                <p className="text-secondary-600">This may take a few moments...</p>
              </div>
            </div>
            <div className="mt-4 bg-gray-100 rounded-full h-2.5 overflow-hidden">
              <div className="bg-gradient-to-r from-primary-500 to-accent-500 h-2.5 rounded-full animate-pulse-gradient" style={{width: '60%'}}></div>
            </div>
          </motion.div>
        )}

        {/* Upload Results */}
        {uploadResults.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="bg-white rounded-xl shadow-lg p-6 mb-8"
          >
            <h3 className="text-xl font-semibold text-secondary-900 mb-5 flex items-center">
              <span className="bg-secondary-100 text-secondary-600 p-2 rounded-lg mr-3">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              </span>
              Processing Results
            </h3>
            
            <div className="space-y-4 max-h-96 overflow-y-auto pr-2 custom-scrollbar">
              {uploadResults.map((result, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className={`p-4 rounded-lg border ${
                    result.success 
                      ? 'bg-green-50 border-green-100' 
                      : 'bg-red-50 border-red-100'
                  }`}
                >
                  <div className="flex items-start">
                    <div className={`p-2 rounded-full mr-3 ${
                      result.success 
                        ? 'bg-green-100 text-green-600' 
                        : 'bg-red-100 text-red-600'
                    }`}>
                      {result.success ? (
                        <CheckCircleIcon className="h-5 w-5" />
                      ) : (
                        <ExclamationCircleIcon className="h-5 w-5" />
                      )}
                    </div>
                    
                    <div className="flex-1">
                      <div className="flex justify-between items-start">
                        <h4 className="font-medium text-secondary-900">
                          {result.filename || 'Document'}
                        </h4>
                        <span className={`text-xs font-semibold px-2 py-1 rounded-full ${
                          result.success 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {result.success ? 'Success' : 'Failed'}
                        </span>
                      </div>
                      
                      <p className={`text-sm mt-1 ${
                        result.success 
                          ? 'text-green-700' 
                          : 'text-red-700'
                      }`}>
                        {result.success 
                          ? `Successfully processed ${result.chunks_created || 0} chunks` 
                          : result.error || 'Processing failed'}
                      </p>
                      
                      {result.success && (
                        <div className="mt-3 flex flex-wrap gap-2">
                          <button className="inline-flex items-center px-3 py-1 bg-white border border-gray-200 rounded-md text-xs font-medium text-secondary-700 hover:bg-gray-50 transition-colors">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 mr-1" viewBox="0 0 20 20" fill="currentColor">
                              <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
                              <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
                            </svg>
                            View Document
                          </button>
                          <button className="inline-flex items-center px-3 py-1 bg-white border border-gray-200 rounded-md text-xs font-medium text-secondary-700 hover:bg-gray-50 transition-colors">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 mr-1" viewBox="0 0 20 20" fill="currentColor">
                              <path d="M7 3a1 1 0 000 2h6a1 1 0 100-2H7zM4 7a1 1 0 011-1h10a1 1 0 110 2H5a1 1 0 01-1-1zM2 11a2 2 0 012-2h12a2 2 0 012 2v4a2 2 0 01-2 2H4a2 2 0 01-2-2v-4z" />
                            </svg>
                            View Chunks
                          </button>
                        </div>
                      )}
                      
                      {!result.success && (
                        <div className="mt-3">
                          <button className="inline-flex items-center px-3 py-1 bg-white border border-gray-200 rounded-md text-xs font-medium text-secondary-700 hover:bg-gray-50 transition-colors">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 mr-1" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
                            </svg>
                            Retry
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}
