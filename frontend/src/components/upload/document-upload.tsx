'use client'

import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useDropzone } from 'react-dropzone'
import { 
  DocumentArrowUpIcon,
  XMarkIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  CogIcon,
  FolderIcon,
  FolderPlusIcon,
  ListBulletIcon,
  Squares2X2Icon,
  EyeIcon,
  PencilIcon,
  TrashIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline'
import { useUploadDocuments, useFrontendDocumentsList } from '@/hooks/use-api'
import toast from 'react-hot-toast'

interface UploadSettings {
  extractContracts: boolean
  chunkSize: number
  enableOCR: boolean
  autoProcess: boolean
}

interface Folder {
  id: string
  name: string
  parentId: string | null
  documents: string[]
  createdAt: Date
}

interface Document {
  id: string
  filename: string
  fileType: string
  uploadedAt: string
  size: string
  status: string
  folderId?: string
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
  const [folders, setFolders] = useState<Folder[]>([])  
  const [currentFolderId, setCurrentFolderId] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [showCreateFolder, setShowCreateFolder] = useState(false)
  const [newFolderName, setNewFolderName] = useState('')
  
  const uploadMutation = useUploadDocuments()
  const { data: documentsData, refetch: refetchDocuments } = useFrontendDocumentsList()

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
      
      // Clear uploaded files and refresh document list
      setUploadedFiles([])
      refetchDocuments()
      
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

  // Folder management functions
  const createFolder = () => {
    if (!newFolderName.trim()) return
    
    const newFolder: Folder = {
      id: Date.now().toString(),
      name: newFolderName.trim(),
      parentId: currentFolderId,
      documents: [],
      createdAt: new Date()
    }
    
    setFolders(prev => [...prev, newFolder])
    setNewFolderName('')
    setShowCreateFolder(false)
    toast.success(`Folder "${newFolder.name}" created successfully`)
  }
  
  const moveDocumentToFolder = (documentId: string, folderId: string | null) => {
    // This would typically make an API call to update the document's folder
    toast.success('Document moved successfully')
  }
  
  const deleteFolder = (folderId: string) => {
    const folder = folders.find(f => f.id === folderId)
    if (folder && folder.documents.length > 0) {
      toast.error('Cannot delete folder with documents. Move documents first.')
      return
    }
    
    setFolders(prev => prev.filter(f => f.id !== folderId))
    if (currentFolderId === folderId) {
      setCurrentFolderId(null)
    }
    toast.success('Folder deleted successfully')
  }

  return (
    <div className="h-full overflow-auto bg-white">
      <div className="max-w-6xl mx-auto p-6">
        <div className="flex justify-between items-center mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h1 className="text-3xl font-bold text-black mb-2">
              Document Upload & Processing
            </h1>
            <p className="text-gray-600">
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
              <h2 className="text-lg font-semibold text-black mb-4">Upload Settings</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <label className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={uploadSettings.extractContracts}
                      onChange={(e) => setUploadSettings(prev => ({ ...prev, extractContracts: e.target.checked }))}
                      className="border-gray-300 text-black focus:ring-gray-400"
                    />
                    <div>
                      <span className="text-sm font-medium text-black">Extract Contract Details</span>
                      <p className="text-xs text-gray-600">Automatically identify and extract contract information</p>
                    </div>
                  </label>
                  
                  <label className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={uploadSettings.enableOCR}
                      onChange={(e) => setUploadSettings(prev => ({ ...prev, enableOCR: e.target.checked }))}
                      className="border-gray-300 text-black focus:ring-gray-400"
                    />
                    <div>
                      <span className="text-sm font-medium text-black">Enable OCR</span>
                      <p className="text-xs text-gray-600">Extract text from scanned documents and images</p>
                    </div>
                  </label>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-black mb-2">
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
                    <p className="text-xs text-gray-600 mt-1">Size of text chunks for processing (500-2000 characters)</p>
                  </div>
                  
                  <label className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={uploadSettings.autoProcess}
                      onChange={(e) => setUploadSettings(prev => ({ ...prev, autoProcess: e.target.checked }))}
                      className="border-gray-300 text-black focus:ring-gray-400"
                    />
                    <div>
                      <span className="text-sm font-medium text-black">Auto-process on Upload</span>
                      <p className="text-xs text-gray-600">Start processing immediately after upload</p>
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
        
        {/* Document Management Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card p-6 mt-6"
        >
          <div className="flex justify-between items-center mb-6">
            <div>
              <h2 className="text-xl font-semibold text-black mb-2">
                Document Library
              </h2>
              <p className="text-gray-600">
                Manage your uploaded documents with folders and organization
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setShowCreateFolder(true)}
                className="btn-ghost flex items-center space-x-2"
              >
                <FolderPlusIcon className="h-4 w-4" />
                <span>New Folder</span>
              </button>
              <div className="border-l border-gray-300 pl-2 ml-2">
                <button
                  onClick={() => setViewMode(viewMode === 'grid' ? 'list' : 'grid')}
                  className="p-2 hover:bg-gray-100 border border-gray-300 transition-colors"
                  title={`Switch to ${viewMode === 'grid' ? 'list' : 'grid'} view`}
                >
                  {viewMode === 'grid' ? (
                    <ListBulletIcon className="h-4 w-4" />
                  ) : (
                    <Squares2X2Icon className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>
          </div>
          
          {/* Create Folder Modal */}
          {showCreateFolder && (
            <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-white rounded-lg max-w-md w-full p-6"
              >
                <h3 className="text-lg font-semibold text-black mb-4">Create New Folder</h3>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Folder Name
                  </label>
                  <input
                    type="text"
                    value={newFolderName}
                    onChange={(e) => setNewFolderName(e.target.value)}
                    className="input-field w-full"
                    placeholder="Enter folder name..."
                    onKeyPress={(e) => e.key === 'Enter' && createFolder()}
                  />
                </div>
                <div className="flex justify-end space-x-2">
                  <button
                    onClick={() => {
                      setShowCreateFolder(false)
                      setNewFolderName('')
                    }}
                    className="btn-ghost"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={createFolder}
                    disabled={!newFolderName.trim()}
                    className="btn-primary disabled:opacity-50"
                  >
                    Create
                  </button>
                </div>
              </motion.div>
            </div>
          )}
          
          {/* Folder Navigation */}
          {currentFolderId && (
            <div className="mb-4 p-3 bg-gray-50 border border-gray-200">
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <button
                  onClick={() => setCurrentFolderId(null)}
                  className="hover:text-black transition-colors"
                >
                  Documents
                </button>
                <span>/</span>
                <span className="text-black font-medium">
                  {folders.find(f => f.id === currentFolderId)?.name || 'Unknown Folder'}
                </span>
              </div>
            </div>
          )}
          
          {/* Folders Grid/List */}
          {folders.filter(f => f.parentId === currentFolderId).length > 0 && (
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-700 mb-3">Folders</h3>
              <div className={viewMode === 'grid' ? 'grid grid-cols-2 md:grid-cols-4 gap-4' : 'space-y-2'}>
                {folders
                  .filter(f => f.parentId === currentFolderId)
                  .map(folder => (
                    <motion.div
                      key={folder.id}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className={`group relative p-3 border border-gray-200 hover:border-gray-300 transition-colors cursor-pointer ${
                        viewMode === 'grid' ? 'text-center' : 'flex items-center justify-between'
                      }`}
                      onClick={() => setCurrentFolderId(folder.id)}
                    >
                      <div className={viewMode === 'grid' ? '' : 'flex items-center space-x-3'}>
                        <FolderIcon className={`text-yellow-500 ${viewMode === 'grid' ? 'h-8 w-8 mx-auto mb-2' : 'h-5 w-5'}`} />
                        <div>
                          <p className="text-sm font-medium text-black">{folder.name}</p>
                          <p className="text-xs text-gray-500">{folder.documents.length} files</p>
                        </div>
                      </div>
                      {viewMode === 'list' && (
                        <div className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              // Edit folder logic
                            }}
                            className="p-1 hover:bg-gray-100 rounded transition-colors"
                          >
                            <PencilIcon className="h-3 w-3 text-gray-500" />
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              deleteFolder(folder.id)
                            }}
                            className="p-1 hover:bg-red-100 rounded transition-colors"
                          >
                            <TrashIcon className="h-3 w-3 text-red-500" />
                          </button>
                        </div>
                      )}
                    </motion.div>
                  ))
                }
              </div>
            </div>
          )}
          
          {/* Documents Grid/List */}
          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-700 mb-3">
              Documents ({documentsData?.documents?.length || 0})
            </h3>
          </div>
          
          {documentsData?.documents?.length > 0 ? (
            <div className={viewMode === 'grid' ? 'grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4' : 'space-y-2'}>
              {documentsData.documents.map((doc: any) => (
                <motion.div
                  key={doc.id}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className={`group relative border border-gray-200 hover:border-gray-300 transition-colors ${
                    viewMode === 'grid' ? 'p-4 text-center' : 'p-3 flex items-center justify-between'
                  }`}
                >
                  <div className={viewMode === 'grid' ? '' : 'flex items-center space-x-3 flex-1'}>
                    <DocumentTextIcon className={`text-blue-500 ${viewMode === 'grid' ? 'h-8 w-8 mx-auto mb-2' : 'h-5 w-5'}`} />
                    <div className={viewMode === 'grid' ? '' : 'flex-1'}>
                      <p className="text-sm font-medium text-black truncate" title={doc.filename}>
                        {doc.filename}
                      </p>
                      <p className="text-xs text-gray-500">
                        {doc.fileType} • {doc.size} • {doc.uploadedAt}
                      </p>
                      <div className={`inline-flex items-center px-2 py-1 text-xs font-medium rounded-full ${
                        doc.status === 'Processed' 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-yellow-100 text-yellow-800'
                      }`}>
                        {doc.status}
                      </div>
                    </div>
                  </div>
                  
                  {viewMode === 'list' && (
                    <div className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={() => {/* Preview document */}}
                        className="p-1 hover:bg-gray-100 rounded transition-colors"
                        title="Preview document"
                      >
                        <EyeIcon className="h-3 w-3 text-gray-500" />
                      </button>
                      <button
                        onClick={() => {/* Move to folder */}}
                        className="p-1 hover:bg-gray-100 rounded transition-colors"
                        title="Move to folder"
                      >
                        <FolderIcon className="h-3 w-3 text-gray-500" />
                      </button>
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12 text-gray-500">
              <DocumentTextIcon className="h-12 w-12 mx-auto mb-4 text-gray-300" />
              <p className="text-lg font-medium mb-2">No documents uploaded yet</p>
              <p className="text-sm">Upload documents using the drag & drop area above</p>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  )
}
