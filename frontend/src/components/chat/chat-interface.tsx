'use client'

import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  PaperAirplaneIcon,
  UserIcon,
  SparklesIcon,
  DocumentArrowUpIcon,
  CogIcon,
  MagnifyingGlassIcon,
  PlusIcon,
  XMarkIcon,
  EyeIcon
} from '@heroicons/react/24/outline'
import { useSendMessage, ChatMessage, useUploadDocuments, useSearchDocuments } from '@/hooks/use-api'
import toast from 'react-hot-toast'
import { useDropzone } from 'react-dropzone'
import DocumentPreview from '../document/document-preview'

export default function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputValue, setInputValue] = useState('')
  const [conversationId] = useState(() => crypto.randomUUID())
  const [showUpload, setShowUpload] = useState(false)
  const [showSearch, setShowSearch] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [previewDocument, setPreviewDocument] = useState<{id: string, filename: string} | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const sendMessage = useSendMessage()
  const uploadDocs = useUploadDocuments()
  const searchDocs = useSearchDocuments()

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async () => {
    if (!inputValue.trim() || sendMessage.isPending) return

    const userMessage: ChatMessage = {
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')

    try {
      const response = await sendMessage.mutateAsync({
        message: userMessage.content,
        conversationId
      })

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.response,
        sources: response.sources,
        timestamp: new Date(response.timestamp),
        message_id: response.message_id
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Chat error:', error)
      toast.error('Failed to send message. Please try again.')
      
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: 'I apologize, but I encountered an error processing your message. Please try again.',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const formatMessage = (content: string) => {
    // Simple formatting for better readability
    return content.split('\n').map((line, index) => (
      <div key={index} className={index > 0 ? 'mt-2' : ''}>
        {line}
      </div>
    ))
  }

  // File upload handling
  const onDrop = async (acceptedFiles: File[]) => {
    try {
      const result = await uploadDocs.mutateAsync(acceptedFiles)
      toast.success(`Uploaded ${acceptedFiles.length} files successfully!`)
      setShowUpload(false)
    } catch (error) {
      toast.error('Failed to upload files')
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt']
    },
    multiple: true
  })

  // Search handling
  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    
    try {
      const result = await searchDocs.mutateAsync({ query: searchQuery })
      toast.success(`Found ${result.documents.length} documents`)
      setShowSearch(false)
      setSearchQuery('')
    } catch (error) {
      toast.error('Search failed')
    }
  }

  return (
    <div className="flex h-full bg-gradient-to-b from-secondary-50/50 to-white">
      {/* Left Sidebar - Tools */}
      <div className="w-16 lg:w-20 bg-white/80 backdrop-blur-sm border-r border-secondary-200/50 flex flex-col items-center py-4 space-y-4">
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => setShowUpload(!showUpload)}
          className={`p-2 lg:p-3 rounded-lg transition-colors ${
            showUpload ? 'bg-primary-100 text-primary-600' : 'hover:bg-secondary-100 text-secondary-600'
          }`}
          title="Upload Documents"
        >
          <DocumentArrowUpIcon className="h-5 w-5 lg:h-6 lg:w-6" />
        </motion.button>
        
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => setShowSearch(!showSearch)}
          className={`p-2 lg:p-3 rounded-lg transition-colors ${
            showSearch ? 'bg-primary-100 text-primary-600' : 'hover:bg-secondary-100 text-secondary-600'
          }`}
          title="Search Documents"
        >
          <MagnifyingGlassIcon className="h-5 w-5 lg:h-6 lg:w-6" />
        </motion.button>
        
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          className="p-2 lg:p-3 rounded-lg hover:bg-secondary-100 text-secondary-600 transition-colors"
          title="Settings"
        >
          <CogIcon className="h-5 w-5 lg:h-6 lg:w-6" />
        </motion.button>
      </div>
      
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Tool Panels */}
        <AnimatePresence>
          {showUpload && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="bg-primary-50 border-b border-primary-200 p-4"
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium text-primary-900">Upload Documents</h3>
                <button onClick={() => setShowUpload(false)}>
                  <XMarkIcon className="h-5 w-5 text-primary-600" />
                </button>
              </div>
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
                  isDragActive
                    ? 'border-primary-400 bg-primary-100'
                    : 'border-primary-300 hover:border-primary-400 hover:bg-primary-50'
                }`}
              >
                <input {...getInputProps()} />
                <DocumentArrowUpIcon className="h-8 w-8 text-primary-500 mx-auto mb-2" />
                <p className="text-sm text-primary-700">
                  {isDragActive ? 'Drop files here...' : 'Drag & drop files or click to browse'}
                </p>
                <p className="text-xs text-primary-600 mt-1">PDF, DOCX, TXT files supported</p>
              </div>
            </motion.div>
          )}
          
          {showSearch && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="bg-accent-50 border-b border-accent-200 p-4"
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium text-accent-900">Search Documents</h3>
                <button onClick={() => setShowSearch(false)}>
                  <XMarkIcon className="h-5 w-5 text-accent-600" />
                </button>
              </div>
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search for documents..."
                  className="flex-1 px-3 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-accent-500"
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                />
                <button
                  onClick={handleSearch}
                  disabled={!searchQuery.trim() || searchDocs.isPending}
                  className="px-4 py-2 bg-accent-600 text-white rounded-lg hover:bg-accent-700 disabled:opacity-50"
                >
                  Search
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        
        {/* Chat Messages */}
        <div className="flex-1 overflow-auto p-4 lg:p-6 space-y-4 lg:space-y-6">
        <AnimatePresence initial={false}>
          {messages.map((message, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
              className={`flex items-start space-x-2 lg:space-x-4 ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {message.role === 'assistant' && (
                <div className="flex-shrink-0 w-8 h-8 lg:w-10 lg:h-10 bg-gradient-to-br from-primary-500 to-accent-500 rounded-full flex items-center justify-center">
                  <SparklesIcon className="w-4 h-4 lg:w-5 lg:h-5 text-white" />
                </div>
              )}

              <div
                className={`max-w-xs lg:max-w-2xl xl:max-w-3xl ${
                  message.role === 'user'
                    ? 'chat-bubble-user text-sm lg:text-base'
                    : 'chat-bubble-assistant text-sm lg:text-base'
                }`}
              >
                <div className="prose prose-sm max-w-none">
                  {formatMessage(message.content)}
                </div>

                {message.sources && message.sources.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-secondary-200">
                    <p className="text-xs text-secondary-600 font-medium mb-2">
                      📚 Sources ({message.sources.length}):
                    </p>
                    <div className="space-y-2">
                      {message.sources.slice(0, 3).map((source, idx) => (
                        <div key={idx} className="text-xs bg-secondary-50 rounded p-2 border border-secondary-200">
                          <div className="flex items-center justify-between mb-1">
                            <div className="font-medium text-secondary-700 truncate flex-1">
                              {source.filename}
                            </div>
                            {source.document_id && (
                              <button
                                onClick={() => setPreviewDocument({
                                  id: source.document_id,
                                  filename: source.filename
                                })}
                                className="ml-2 px-2 py-1 text-xs bg-primary-100 hover:bg-primary-200 text-primary-700 rounded transition-colors flex items-center space-x-1"
                                title="Preview document"
                              >
                                <EyeIcon className="h-3 w-3" />
                                <span>View</span>
                              </button>
                            )}
                          </div>
                          <div className="text-secondary-500 leading-relaxed">
                            {source.chunk_text?.substring(0, 120)}...
                          </div>
                          {source.similarity && (
                            <div className="mt-1 text-xs text-secondary-400">
                              Relevance: {Math.round(source.similarity * 100)}%
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="mt-2 text-xs text-secondary-400">
                  {message.timestamp?.toLocaleTimeString()}
                </div>
              </div>

              {message.role === 'user' && (
                <div className="flex-shrink-0 w-8 h-8 lg:w-10 lg:h-10 bg-secondary-200 rounded-full flex items-center justify-center">
                  <UserIcon className="w-4 h-4 lg:w-5 lg:h-5 text-secondary-600" />
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Loading indicator */}
        {sendMessage.isPending && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-start space-x-2 lg:space-x-4"
          >
            <div className="flex-shrink-0 w-8 h-8 lg:w-10 lg:h-10 bg-gradient-to-br from-primary-500 to-accent-500 rounded-full flex items-center justify-center">
              <SparklesIcon className="w-4 h-4 lg:w-5 lg:h-5 text-white animate-pulse" />
            </div>
            <div className="chat-bubble-assistant text-sm lg:text-base">
              <div className="loading-dots text-secondary-500">
                <div></div>
                <div></div>
                <div></div>
              </div>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
        </div>

        {/* Input Area - Compact */}
        <div className="border-t border-secondary-200 bg-white/80 backdrop-blur-sm p-3 lg:p-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex items-end space-x-2 lg:space-x-3">
              <div className="flex-1">
                <textarea
                  ref={inputRef}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask me anything about your contracts..."
                  className="input-field resize-none min-h-[44px] lg:min-h-[52px] max-h-32 py-3 lg:py-4 text-sm lg:text-base"
                  rows={1}
                  disabled={sendMessage.isPending}
                />
              </div>
              <motion.button
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || sendMessage.isPending}
                className="btn-primary p-3 lg:p-4 disabled:opacity-50 disabled:cursor-not-allowed"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <PaperAirplaneIcon className="w-4 h-4 lg:w-5 lg:h-5" />
              </motion.button>
            </div>
            
            <div className="mt-2 text-xs text-secondary-500 text-center hidden lg:block">
              Press Enter to send, Shift+Enter for new line
            </div>
          </div>
        </div>
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
