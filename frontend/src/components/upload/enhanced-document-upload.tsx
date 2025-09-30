'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { DocumentPlusIcon, EyeIcon, Cog6ToothIcon, CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/outline'
import { toast } from 'react-hot-toast'

interface UploadResult {
  filename: string
  success: boolean
  document_id?: string
  chunks_created?: number
  contract_extracted?: boolean
  visual_content_found?: boolean
  visual_elements?: number
  processing_method?: string
  error?: string
}

interface EnhancedUploadResponse {
  results: UploadResult[]
  processing_type: string
  vision_enabled: boolean
  total_files: number
  successful_files: number
}

export default function EnhancedDocumentUpload() {
  const [files, setFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [results, setResults] = useState<UploadResult[]>([])
  const [useVision, setUseVision] = useState(true)
  const [customChunkSize, setCustomChunkSize] = useState<number>(1000)
  const [processingProgress, setProcessingProgress] = useState<Record<string, any>>({})

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles(prev => [...prev, ...acceptedFiles])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/gif': ['.gif'],
      'image/bmp': ['.bmp'],
      'image/tiff': ['.tiff']
    },
    multiple: true,
    maxSize: 50 * 1024 * 1024 // 50MB max
  })

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }

  const uploadFiles = async () => {
    if (files.length === 0) {
      toast.error('Please select files to upload')
      return
    }

    setUploading(true)
    setResults([])
    setProcessingProgress({})

    try {
      const formData = new FormData()
      files.forEach(file => {
        formData.append('files', file)
      })

      // Add processing options
      formData.append('use_vision', useVision.toString())
      if (customChunkSize !== 1000) {
        formData.append('custom_chunk_size', customChunkSize.toString())
      }

      const response = await fetch('/api/upload-enhanced', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`)
      }

      const data: EnhancedUploadResponse = await response.json()
      
      setResults(data.results)
      
      // Show summary toast
      toast.success(
        `Enhanced processing completed: ${data.successful_files}/${data.total_files} files processed successfully`,
        { duration: 5000 }
      )

      // Show detailed results for each file
      data.results.forEach((result) => {
        if (result.success) {
          let message = `✅ ${result.filename}: ${result.chunks_created} chunks created`
          
          if (result.visual_elements && result.visual_elements > 0) {
            message += `, ${result.visual_elements} visual elements analyzed`
          }
          
          if (result.contract_extracted) {
            message += `, contract data extracted`
          }
          
          toast.success(message, { duration: 4000 })
        } else {
          toast.error(`❌ ${result.filename}: ${result.error}`, { duration: 6000 })
        }
      })

      // Clear files on successful upload
      if (data.successful_files > 0) {
        setFiles([])
      }

    } catch (error) {
      console.error('Enhanced upload error:', error)
      toast.error(error instanceof Error ? error.message : 'Enhanced upload failed')
    } finally {
      setUploading(false)
    }
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const getFileTypeIcon = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase()
    if (['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'].includes(ext || '')) {
      return '🖼️'
    } else if (ext === 'pdf') {
      return '📄'
    } else if (ext === 'docx') {
      return '📝'
    }
    return '📄'
  }

  const getProcessingMethodBadge = (method?: string) => {
    switch (method) {
      case 'enhanced_with_vision':
        return <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
          <EyeIcon className="w-3 h-3 mr-1" />
          Vision API
        </span>
      case 'enhanced_no_vision':
        return <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
          <Cog6ToothIcon className="w-3 h-3 mr-1" />
          Enhanced
        </span>
      default:
        return null
    }
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Enhanced Document Upload</h3>
        <p className="text-sm text-gray-600">
          Upload documents with AI-powered visual analysis and large document support
        </p>
      </div>

      {/* Processing Options */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-medium text-gray-900 mb-3">Processing Options</h4>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Vision API Toggle */}
          <div className="flex items-center justify-between">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Enable Vision API
              </label>
              <p className="text-xs text-gray-500">
                Extract text from images and analyze visual content
              </p>
            </div>
            <button
              type="button"
              onClick={() => setUseVision(!useVision)}
              className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 ${
                useVision ? 'bg-purple-600' : 'bg-gray-200'
              }`}
            >
              <span
                className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                  useVision ? 'translate-x-5' : 'translate-x-0'
                }`}
              />
            </button>
          </div>

          {/* Custom Chunk Size */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Chunk Size
            </label>
            <select
              value={customChunkSize}
              onChange={(e) => setCustomChunkSize(Number(e.target.value))}
              className="block w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 sm:text-sm"
            >
              <option value={500}>Small (500 chars)</option>
              <option value={1000}>Default (1000 chars)</option>
              <option value={1500}>Large (1500 chars)</option>
              <option value={2000}>Extra Large (2000 chars)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Drop Zone */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
          isDragActive
            ? 'border-purple-500 bg-purple-50'
            : 'border-gray-300 hover:border-purple-400 hover:bg-gray-50'
        }`}
      >
        <input {...getInputProps()} />
        <DocumentPlusIcon className="mx-auto h-12 w-12 text-gray-400" />
        <p className="mt-2 text-sm text-gray-600">
          {isDragActive
            ? 'Drop the files here...'
            : 'Drop files here, or click to select'}
        </p>
        <p className="text-xs text-gray-500 mt-1">
          Supports: PDF, DOCX, TXT, PNG, JPG, GIF, BMP, TIFF (up to 50MB each)
        </p>
      </div>

      {/* Selected Files */}
      {files.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-medium text-gray-900 mb-2">
            Selected Files ({files.length})
          </h4>
          <div className="space-y-2">
            {files.map((file, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-2 bg-gray-50 rounded-md"
              >
                <div className="flex items-center">
                  <span className="text-lg mr-2">{getFileTypeIcon(file.name)}</span>
                  <div>
                    <p className="text-sm font-medium text-gray-900">{file.name}</p>
                    <p className="text-xs text-gray-500">{formatFileSize(file.size)}</p>
                  </div>
                </div>
                <button
                  onClick={() => removeFile(index)}
                  className="text-red-600 hover:text-red-800"
                >
                  <XCircleIcon className="h-5 w-5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Upload Button */}
      <div className="mt-6">
        <button
          onClick={uploadFiles}
          disabled={uploading || files.length === 0}
          className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-purple-600 hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          {uploading ? (
            <>
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Processing with Enhanced AI...
            </>
          ) : (
            <>
              <EyeIcon className="w-5 h-5 mr-2" />
              Upload & Process with Enhanced AI
            </>
          )}
        </button>
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div className="mt-6">
          <h4 className="text-sm font-medium text-gray-900 mb-3">Processing Results</h4>
          <div className="space-y-3">
            {results.map((result, index) => (
              <div
                key={index}
                className={`p-4 rounded-lg border ${
                  result.success ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start">
                    {result.success ? (
                      <CheckCircleIcon className="h-5 w-5 text-green-600 mt-0.5 mr-2 flex-shrink-0" />
                    ) : (
                      <XCircleIcon className="h-5 w-5 text-red-600 mt-0.5 mr-2 flex-shrink-0" />
                    )}
                    <div>
                      <p className="text-sm font-medium text-gray-900">{result.filename}</p>
                      {result.success ? (
                        <div className="mt-1 text-xs text-gray-600">
                          <p>📄 {result.chunks_created} text chunks created</p>
                          {result.visual_elements && result.visual_elements > 0 && (
                            <p>🖼️ {result.visual_elements} visual elements analyzed</p>
                          )}
                          {result.contract_extracted && (
                            <p>📋 Contract data extracted successfully</p>
                          )}
                        </div>
                      ) : (
                        <p className="mt-1 text-xs text-red-600">{result.error}</p>
                      )}
                    </div>
                  </div>
                  <div className="ml-4">
                    {getProcessingMethodBadge(result.processing_method)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}