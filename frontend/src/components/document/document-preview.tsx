'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  DocumentTextIcon,
  XMarkIcon,
  ArrowDownTrayIcon,
  EyeIcon,
  DocumentIcon
} from '@heroicons/react/24/outline'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'
import toast from 'react-hot-toast'

interface DocumentPreviewProps {
  documentId: string
  filename: string
  isOpen: boolean
  onClose: () => void
}

interface DocumentData {
  id: string
  filename: string
  file_type: string
  content: string
  created_at: string
  metadata: Record<string, any>
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function DocumentPreview({ documentId, filename, isOpen, onClose }: DocumentPreviewProps) {
  const [isDownloading, setIsDownloading] = useState(false)

  // Fetch document data
  const { data: document, isLoading, error } = useQuery<DocumentData>({
    queryKey: ['document', documentId],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE_URL}/documents/${documentId}`)
      return response.data
    },
    enabled: isOpen && !!documentId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  })

  const handleDownload = async () => {
    if (!documentId) return
    
    setIsDownloading(true)
    try {
      const response = await axios.get(
        `${API_BASE_URL}/documents/${documentId}/download`,
        { responseType: 'blob' }
      )
      
      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', filename)
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
      
      toast.success('Document downloaded successfully!')
    } catch (error) {
      console.error('Download failed:', error)
      toast.error('Failed to download document')
    } finally {
      setIsDownloading(false)
    }
  }

  const formatContent = (content: string) => {
    // Simple formatting for better readability
    const lines = content.split('\n')
    return lines.map((line, index) => (
      <p key={index} className={`${line.trim() === '' ? 'mb-4' : 'mb-2'} text-sm leading-relaxed`}>
        {line.trim() || '\u00A0'} {/* Non-breaking space for empty lines */}
      </p>
    ))
  }

  const getFileIcon = (fileType: string) => {
    switch (fileType?.toLowerCase()) {
      case 'pdf':
        return <DocumentIcon className="h-6 w-6 text-red-500" />
      case 'docx':
      case 'doc':
        return <DocumentTextIcon className="h-6 w-6 text-blue-500" />
      default:
        return <DocumentTextIcon className="h-6 w-6 text-gray-500" />
    }
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
            onClick={onClose}
          />
          
          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="fixed inset-4 md:inset-8 lg:inset-16 bg-white rounded-lg shadow-2xl z-50 flex flex-col"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 lg:p-6 border-b border-gray-200">
              <div className="flex items-center space-x-3">
                {document && getFileIcon(document.file_type)}
                <div>
                  <h2 className="text-lg lg:text-xl font-semibold text-gray-900 truncate max-w-md">
                    {filename}
                  </h2>
                  {document && (
                    <p className="text-sm text-gray-500">
                      {document.file_type?.toUpperCase()} • {new Date(document.created_at).toLocaleDateString()}
                    </p>
                  )}
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleDownload}
                  disabled={isDownloading || !document}
                  className="btn-ghost flex items-center space-x-2"
                >
                  <ArrowDownTrayIcon className="h-5 w-5" />
                  <span className="hidden sm:inline">
                    {isDownloading ? 'Downloading...' : 'Download'}
                  </span>
                </motion.button>
                
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={onClose}
                  className="p-2 hover:bg-gray-100 rounded-lg"
                >
                  <XMarkIcon className="h-5 w-5 text-gray-500" />
                </motion.button>
              </div>
            </div>
            
            {/* Content */}
            <div className="flex-1 overflow-auto">
              {isLoading && (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500 mx-auto mb-4"></div>
                    <p className="text-gray-600">Loading document...</p>
                  </div>
                </div>
              )}
              
              {error && (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center text-red-600">
                    <p className="mb-2">Failed to load document</p>
                    <button 
                      onClick={() => window.location.reload()}
                      className="text-sm underline hover:no-underline"
                    >
                      Try again
                    </button>
                  </div>
                </div>
              )}
              
              {document && (
                <div className="p-4 lg:p-6">
                  {/* Document metadata */}
                  {document.metadata && Object.keys(document.metadata).length > 0 && (
                    <div className="mb-6 p-4 bg-gray-50 rounded-lg">
                      <h3 className="text-sm font-medium text-gray-900 mb-2">Document Information</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
                        {Object.entries(document.metadata).map(([key, value]) => (
                          <div key={key} className="flex">
                            <span className="font-medium text-gray-600 capitalize">{key.replace('_', ' ')}:</span>
                            <span className="ml-2 text-gray-900">{String(value)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Document content */}
                  <div className="prose prose-sm max-w-none">
                    <div className="bg-white border border-gray-200 rounded-lg p-4 lg:p-6 font-mono text-sm leading-relaxed">
                      {formatContent(document.content)}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}