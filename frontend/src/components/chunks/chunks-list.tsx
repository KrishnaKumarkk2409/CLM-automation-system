'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  DocumentTextIcon, 
  MagnifyingGlassIcon,
  EyeIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  HashtagIcon,
  XMarkIcon
} from '@heroicons/react/24/outline'
import toast from 'react-hot-toast'

interface Chunk {
  id: string
  document_id: string
  document_name: string
  chunk_index: number
  text_preview: string
  full_text: string
  created_at: string
  token_count: number
}

interface ChunksListProps {
  onBack?: () => void
}

export default function ChunksList({ onBack }: ChunksListProps) {
  const [chunks, setChunks] = useState<Chunk[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedChunk, setSelectedChunk] = useState<Chunk | null>(null)
  const [currentPage, setCurrentPage] = useState(1)
  const [itemsPerPage] = useState(20)

  useEffect(() => {
    fetchChunks()
  }, [])

  const fetchChunks = async () => {
    try {
      const response = await fetch('http://localhost:8000/chunks')
      if (!response.ok) throw new Error('Failed to fetch chunks')
      
      const data = await response.json()
      setChunks(data.chunks || [])
    } catch (error) {
      console.error('Error fetching chunks:', error)
      toast.error('Failed to load text chunks')
    } finally {
      setLoading(false)
    }
  }

  const filteredChunks = chunks.filter(chunk => {
    if (!searchQuery) return true
    return (
      chunk.document_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      chunk.text_preview.toLowerCase().includes(searchQuery.toLowerCase()) ||
      chunk.full_text.toLowerCase().includes(searchQuery.toLowerCase())
    )
  })

  const totalPages = Math.ceil(filteredChunks.length / itemsPerPage)
  const startIndex = (currentPage - 1) * itemsPerPage
  const endIndex = startIndex + itemsPerPage
  const currentChunks = filteredChunks.slice(startIndex, endIndex)

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    })
  }

  const getTokenColor = (tokenCount: number) => {
    if (tokenCount < 50) return 'text-green-600 bg-green-50'
    if (tokenCount < 100) return 'text-yellow-600 bg-yellow-50'
    if (tokenCount < 150) return 'text-orange-600 bg-orange-50'
    return 'text-red-600 bg-red-50'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-black"></div>
      </div>
    )
  }

  return (
    <div className="p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-4">
          {onBack && (
            <button
              onClick={onBack}
              className="p-2 hover:bg-gray-100 border border-gray-300 transition-colors"
            >
              <ChevronLeftIcon className="h-5 w-5" />
            </button>
          )}
          <div>
            <h1 className="text-2xl font-bold text-gray-900 flex items-center">
              <DocumentTextIcon className="h-8 w-8 mr-3 text-blue-600" />
              Text Chunks
            </h1>
            <p className="text-gray-600">
              {filteredChunks.length} chunks • Total tokens: {chunks.reduce((sum, chunk) => sum + chunk.token_count, 0).toLocaleString()}
            </p>
          </div>
        </div>
      </div>

      {/* Search */}
      <div className="relative mb-6">
        <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
        <input
          type="text"
          placeholder="Search chunks by content or document name..."
          value={searchQuery}
          onChange={(e) => {
            setSearchQuery(e.target.value)
            setCurrentPage(1) // Reset to first page when searching
          }}
          className="w-full pl-10 pr-4 py-3 border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      </div>

      {/* Chunks List */}
      <div className="space-y-4 mb-6">
        {currentChunks.map((chunk, index) => (
          <motion.div
            key={chunk.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            className="bg-white border border-gray-200 hover:border-gray-300 transition-colors"
          >
            <div className="p-4">
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <h3 className="text-lg font-medium text-gray-900 mb-1">
                    {chunk.document_name}
                  </h3>
                  <div className="flex items-center space-x-4 text-sm text-gray-500">
                    <span className="flex items-center">
                      <HashtagIcon className="h-4 w-4 mr-1" />
                      Chunk {chunk.chunk_index}
                    </span>
                    <span>{formatDate(chunk.created_at)}</span>
                    <span className={`px-2 py-1 text-xs font-medium rounded ${getTokenColor(chunk.token_count)}`}>
                      {chunk.token_count} tokens
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedChunk(chunk)}
                  className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 border border-gray-200 transition-colors"
                  title="View full text"
                >
                  <EyeIcon className="h-5 w-5" />
                </button>
              </div>
              
              <div className="text-gray-700 bg-gray-50 p-3 text-sm leading-relaxed font-mono">
                {chunk.text_preview}
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between border-t border-gray-200 pt-4">
          <div className="text-sm text-gray-500">
            Showing {startIndex + 1} to {Math.min(endIndex, filteredChunks.length)} of {filteredChunks.length} chunks
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
              disabled={currentPage === 1}
              className="px-3 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronLeftIcon className="h-4 w-4" />
            </button>
            
            <div className="flex items-center space-x-1">
              {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                const page = i + Math.max(1, currentPage - 2)
                if (page > totalPages) return null
                
                return (
                  <button
                    key={page}
                    onClick={() => setCurrentPage(page)}
                    className={`px-3 py-2 text-sm font-medium ${
                      currentPage === page
                        ? 'text-blue-600 bg-blue-50 border-blue-500'
                        : 'text-gray-500 bg-white border-gray-300 hover:bg-gray-50'
                    } border`}
                  >
                    {page}
                  </button>
                )
              })}
            </div>
            
            <button
              onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
              disabled={currentPage === totalPages}
              className="px-3 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronRightIcon className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      {/* Full Text Modal */}
      {selectedChunk && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="bg-white rounded-lg max-w-4xl w-full max-h-[80vh] overflow-hidden"
          >
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">
                    {selectedChunk.document_name}
                  </h2>
                  <p className="text-sm text-gray-500">
                    Chunk {selectedChunk.chunk_index} • {selectedChunk.token_count} tokens
                  </p>
                </div>
                <button
                  onClick={() => setSelectedChunk(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <XMarkIcon className="h-6 w-6" />
                </button>
              </div>
            </div>
            
            <div className="p-6 overflow-y-auto max-h-96">
              <pre className="text-sm text-gray-700 whitespace-pre-wrap font-mono leading-relaxed bg-gray-50 p-4 rounded">
                {selectedChunk.full_text}
              </pre>
            </div>
          </motion.div>
        </div>
      )}

      {/* Empty State */}
      {filteredChunks.length === 0 && !loading && (
        <div className="text-center py-12">
          <DocumentTextIcon className="h-12 w-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            {searchQuery ? 'No chunks found' : 'No text chunks available'}
          </h3>
          <p className="text-gray-500">
            {searchQuery 
              ? 'Try adjusting your search query' 
              : 'Upload some documents to see text chunks here'}
          </p>
          {searchQuery && (
            <button
              onClick={() => {
                setSearchQuery('')
                setCurrentPage(1)
              }}
              className="mt-4 px-4 py-2 text-sm font-medium text-blue-600 hover:text-blue-800"
            >
              Clear search
            </button>
          )}
        </div>
      )}
    </div>
  )
}
