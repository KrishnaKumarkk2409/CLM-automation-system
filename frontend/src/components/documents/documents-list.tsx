'use client'

import { motion } from 'framer-motion'
import { 
  DocumentTextIcon,
  EyeIcon,
  ArrowDownTrayIcon,
  CalendarIcon
} from '@heroicons/react/24/outline'
import { useDocumentsList } from '@/hooks/use-api'
import { useState } from 'react'
import DocumentPreview from '../document/document-preview'

export default function DocumentsList() {
  const { data, isLoading, error } = useDocumentsList()
  const [previewDocument, setPreviewDocument] = useState<{id: string, filename: string} | null>(null)
  
  const documents = data?.documents || []

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Processed':
        return 'text-green-600 bg-green-50'
      case 'Processing':
        return 'text-yellow-600 bg-yellow-50'
      case 'Failed':
        return 'text-red-600 bg-red-50'
      default:
        return 'text-gray-600 bg-gray-50'
    }
  }

  return (
    <div className="h-full overflow-auto bg-white p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-black mb-2">Documents</h1>
          <p className="text-gray-600">View and manage your uploaded documents</p>
        </motion.div>

        {/* Documents Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card"
        >
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Document
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Size
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Uploaded
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {documents.map((doc: any, index: number) => (
                  <motion.tr
                    key={doc.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 + index * 0.05 }}
                    className="hover:bg-gray-50"
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <DocumentTextIcon className="h-5 w-5 text-gray-400 mr-3" />
                        <div>
                          <div className="text-sm font-medium text-black">{doc.filename}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="px-2 py-1 text-xs font-medium bg-gray-100 text-gray-800 border border-gray-200">
                        {doc.fileType}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                      {doc.size}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center text-sm text-gray-600">
                        <CalendarIcon className="h-4 w-4 mr-1" />
                        {doc.uploadedAt}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 text-xs font-medium border ${getStatusColor(doc.status)}`}>
                        {doc.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <div className="flex space-x-2">
                        <button
                          onClick={() => setPreviewDocument({
                            id: doc.id,
                            filename: doc.filename
                          })}
                          className="p-1 text-gray-400 hover:text-black border border-gray-300 hover:border-gray-400 bg-white"
                          title="Preview"
                        >
                          <EyeIcon className="h-4 w-4" />
                        </button>
                        <button
                          onClick={() => {
                            // Trigger download
                            window.open(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/documents/${doc.id}/download`, '_blank')
                          }}
                          className="p-1 text-gray-400 hover:text-black border border-gray-300 hover:border-gray-400 bg-white"
                          title="Download"
                        >
                          <ArrowDownTrayIcon className="h-4 w-4" />
                        </button>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>

          {isLoading && (
            <div className="text-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-black mx-auto mb-4"></div>
              <p className="text-gray-500">Loading documents...</p>
            </div>
          )}
          
          {error && (
            <div className="text-center py-12">
              <DocumentTextIcon className="h-12 w-12 text-red-300 mx-auto mb-4" />
              <p className="text-red-500">Failed to load documents</p>
            </div>
          )}
          
          {!isLoading && !error && documents.length === 0 && (
            <div className="text-center py-12">
              <DocumentTextIcon className="h-12 w-12 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500">No documents uploaded yet</p>
            </div>
          )}
        </motion.div>
      </div>
      
      {/* Document Preview Modal */}
      {previewDocument && (
        <DocumentPreview
          documentId={previewDocument.id}
          filename={previewDocument.filename}
          isOpen={!!previewDocument}
          onClose={() => setPreviewDocument(null)}
        />
      )}
    </div>
  )
}
