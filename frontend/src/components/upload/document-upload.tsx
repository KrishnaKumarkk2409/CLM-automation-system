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
        <div className="flex justify-between items-center mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h1 className="text-3xl font-bold text-secondary-900 mb-2">
              Document Upload & Processing
            </h1>
            <p className="text-secondary-600">
              Upload contracts and documents for AI analysis
            </p>
          </motion.div>
          
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            onClick={() => setShowSettings(!showSettings)}
            className="btn-ghost flex items-center space-x-2"
          >
            <CogIcon className="h-5 w-5" />
            <span>Settings</span>
          </motion.button>
        </div>

        {/* Upload Settings */}
        <AnimatePresence>
          {showSettings && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="card p-6 mb-6 overflow-hidden"
            >
              <h2 className="text-lg font-semibold text-secondary-900 mb-4">Upload Settings</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <label className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={uploadSettings.extractContracts}
                      onChange={(e) => setUploadSettings(prev => ({ ...prev, extractContracts: e.target.checked }))}
                      className="rounded border-secondary-300 text-primary-600 focus:ring-primary-500"
                    />
                    <div>
                      <span className="text-sm font-medium text-secondary-900">Extract Contract Details</span>
                      <p className="text-xs text-secondary-600">Automatically identify and extract contract information</p>
                    </div>
                  </label>
                  
                  <label className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={uploadSettings.enableOCR}
                      onChange={(e) => setUploadSettings(prev => ({ ...prev, enableOCR: e.target.checked }))}
                      className="rounded border-secondary-300 text-primary-600 focus:ring-primary-500"
                    />
                    <div>
                      <span className="text-sm font-medium text-secondary-900">Enable OCR</span>
                      <p className="text-xs text-secondary-600">Extract text from scanned documents and images</p>
                    </div>
                  </label>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-secondary-900 mb-2">
                      Chunk Size: {uploadSettings.chunkSize}
                    </label>
                    <input
                      type="range"
                      min="500"
                      max="2000"
                      step="100"
                      value={uploadSettings.chunkSize}
                      onChange={(e) => setUploadSettings(prev => ({ ...prev, chunkSize: parseInt(e.target.value) }))}
                      className="w-full"
                    />
                    <p className="text-xs text-secondary-600 mt-1">Size of text chunks for processing (500-2000 characters)</p>
                  </div>
                  
                  <label className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={uploadSettings.autoProcess}
                      onChange={(e) => setUploadSettings(prev => ({ ...prev, autoProcess: e.target.checked }))}
                      className="rounded border-secondary-300 text-primary-600 focus:ring-primary-500"
                    />
                    <div>
                      <span className="text-sm font-medium text-secondary-900">Auto-process on Upload</span>
                      <p className="text-xs text-secondary-600">Start processing immediately after upload</p>
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
          className="card mb-6"
        >
          <div
            {...getRootProps()}
            className={`p-8 border-2 border-dashed rounded-lg transition-all cursor-pointer ${
              isDragActive
                ? 'border-primary-400 bg-primary-50'
                : 'border-secondary-300 hover:border-primary-400 hover:bg-primary-50/50'
            }`}
          >
            <input {...getInputProps()} />
            <div className="text-center">
              <DocumentArrowUpIcon className="w-12 h-12 text-primary-500 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-secondary-900 mb-2">
                {isDragActive ? 'Drop files here...' : 'Upload Documents'}
              </h3>
              <p className="text-secondary-600 mb-4">
                Drag & drop files here, or click to browse
              </p>
              <p className="text-sm text-secondary-500">
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
            className="card p-6 mb-6"
          >
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-secondary-900">
                Files to Upload ({uploadedFiles.length})
              </h3>
              <div className="flex space-x-2">
                {!uploadSettings.autoProcess && (
                  <button
                    onClick={() => handleUpload()}
                    disabled={isProcessing}
                    className="btn-primary disabled:opacity-50"
                  >
                    {isProcessing ? 'Processing...' : 'Upload All'}
                  </button>
                )}
                <button onClick={clearAll} className="btn-ghost">
                  Clear All
                </button>
              </div>
            </div>
            
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {uploadedFiles.map((file, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-secondary-50 rounded-lg">
                  <div className="flex-1">
                    <p className="text-sm font-medium text-secondary-900">{file.name}</p>
                    <p className="text-xs text-secondary-600">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                  <button
                    onClick={() => removeFile(index)}
                    className="p-1 hover:bg-secondary-200 rounded"
                  >
                    <XMarkIcon className="h-4 w-4 text-secondary-600" />
                  </button>
                </div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Processing Status */}
        {isProcessing && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="card p-6 mb-6"
          >
            <div className="flex items-center space-x-3">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-500"></div>
              <span className="text-secondary-900 font-medium">Processing documents...</span>
            </div>
            <div className="mt-4 bg-secondary-200 rounded-full h-2">
              <div className="bg-primary-500 h-2 rounded-full animate-pulse" style={{width: '60%'}}></div>
            </div>
          </motion.div>
        )}

        {/* Upload Results */}
        {uploadResults.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="card p-6"
          >
            <h3 className="text-lg font-semibold text-secondary-900 mb-4">
              Upload Results
            </h3>
            
            <div className="space-y-3">
              {uploadResults.map((result, index) => (
                <div key={index} className={`p-4 rounded-lg border ${
                  result.success ? 'bg-success-50 border-success-200' : 'bg-danger-50 border-danger-200'
                }`}>
                  <div className="flex items-start space-x-3">
                    {result.success ? (
                      <CheckCircleIcon className="h-5 w-5 text-success-600 flex-shrink-0 mt-0.5" />
                    ) : (
                      <ExclamationCircleIcon className="h-5 w-5 text-danger-600 flex-shrink-0 mt-0.5" />
                    )}
                    <div className="flex-1">
                      <p className="font-medium text-secondary-900">{result.filename}</p>
                      {result.success ? (
                        <div className="text-sm text-secondary-600 mt-1">
                          <p>✅ Document processed successfully</p>
                          <p>📄 {result.chunks_created || 0} text chunks created</p>
                          {result.contract_extracted && <p>📋 Contract details extracted</p>}
                        </div>
                      ) : (
                        <p className="text-sm text-danger-700 mt-1">
                          ❌ {result.error || 'Processing failed'}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}
