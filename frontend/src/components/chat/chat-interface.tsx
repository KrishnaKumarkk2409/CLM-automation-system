'use client'

import { useState, useRef, useEffect, useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeKatex from 'rehype-katex'
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
  EyeIcon,
  PaperClipIcon,
  Bars3BottomLeftIcon
} from '@heroicons/react/24/outline'
import { useSendMessage, ChatMessage, useUploadDocuments, useSearchDocuments, useChatHistory } from '@/hooks/use-api'
import toast from 'react-hot-toast'
import { useDropzone } from 'react-dropzone'
import DocumentPreview from '../document/document-preview'

export default function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputValue, setInputValue] = useState('')
  const [conversationId, setConversationId] = useState(() => {
    // Browser-compatible UUID generator
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0
      const v = c == 'x' ? r : (r & 0x3 | 0x8)
      return v.toString(16)
    })
  })
  const [historyOpen, setHistoryOpen] = useState(true)
  const [showUpload, setShowUpload] = useState(false)
  const [showSearch, setShowSearch] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [previewDocument, setPreviewDocument] = useState<{id: string, filename: string} | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const sendMessage = useSendMessage()
  const uploadDocs = useUploadDocuments()
  const searchDocs = useSearchDocuments()
  const { data: historyData } = useChatHistory()

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Load messages for selected conversation from history
  useEffect(() => {
    if (!historyData?.messages) return
    if (!conversationId) return
    const convMessages = historyData.messages
      .filter(m => m.conversation_id === conversationId)
      .map(m => ({
        role: m.role,
        content: m.content,
        timestamp: new Date(m.created_at)
      })) as ChatMessage[]
    if (convMessages.length) {
      setMessages(convMessages)
    }
  }, [historyData, conversationId])

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

  const handleAttachClick = () => {
    fileInputRef.current?.click()
  }

  const handleFilesSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length === 0) return
    const docTypes = ['application/pdf', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
    const hasProcessable = files.some(f => docTypes.includes(f.type))
    if (hasProcessable) {
      const confirmProcess = window.confirm('Do you also want to process these file(s) into the database for future use?')
      try {
        await uploadDocs.mutateAsync(files)
        toast.success('Files uploaded successfully')
        if (confirmProcess) {
          toast.success('Processing has been requested and will be reflected shortly.')
        }
      } catch (e) {
        toast.error('Upload failed')
      }
    } else {
      try {
        await uploadDocs.mutateAsync(files)
        toast.success('Attachment uploaded')
      } catch {
        toast.error('Attachment upload failed')
      }
    }
    // reset
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

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

  // Build conversation list from history
  const conversations = useMemo(() => {
    const map: Record<string, { id: string; last: string; count: number; updatedAt: string }> = {}
    (historyData?.messages || []).forEach(m => {
      const item = map[m.conversation_id] || { id: m.conversation_id, last: '', count: 0, updatedAt: m.created_at }
      item.count += 1
      item.last = m.content
      item.updatedAt = m.created_at
      map[m.conversation_id] = item
    })
    return Object.values(map).sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime())
  }, [historyData]);

  return (
    <div className="h-full bg-white">
      <div className="flex h-full">
        {/* History Sidebar */}
        <div className={`border-r border-gray-200 bg-gray-50 ${historyOpen ? 'w-64' : 'w-0'} overflow-hidden transition-all duration-200 hidden md:block`}>
          <div className="p-3 border-b flex items-center justify-between">
            <div className="font-medium text-sm text-black">Chat History</div>
            <button className="text-xs border px-2 py-1" onClick={() => {
              // Start new chat
              const id = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                const r = Math.random() * 16 | 0
                const v = c == 'x' ? r : (r & 0x3 | 0x8)
                return v.toString(16)
              })
              setConversationId(id)
              setMessages([])
            }}>New</button>
          </div>
          <div className="overflow-auto h-[calc(100%-48px)]">
            {conversations.map(conv => (
              <button key={conv.id} onClick={() => setConversationId(conv.id)} className={`w-full text-left p-3 border-b hover:bg-white ${conversationId === conv.id ? 'bg-white' : ''}`}>
                <div className="text-xs text-gray-500 truncate">{conv.id}</div>
                <div className="text-sm text-black truncate mt-1">{conv.last}</div>
                <div className="text-xs text-gray-400 mt-1">{new Date(conv.updatedAt).toLocaleString()}</div>
              </button>
            ))}
            {conversations.length === 0 && (
              <div className="p-4 text-xs text-gray-500">No messages yet</div>
            )}
          </div>
        </div>
      
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col h-full">
        <div className="border-b p-2 flex items-center justify-between md:hidden">
          <button onClick={() => setHistoryOpen(!historyOpen)} className="border px-2 py-1 text-sm flex items-center space-x-1">
            <Bars3BottomLeftIcon className="h-4 w-4" />
            <span>History</span>
          </button>
        </div>
        
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
                <div className="flex-shrink-0 w-8 h-8 lg:w-10 lg:h-10 bg-black border border-black flex items-center justify-center">
                  <SparklesIcon className="w-4 h-4 lg:w-5 lg:h-5 text-white" />
                </div>
              )}

              <div
                className={`max-w-xs lg:max-w-2xl xl:max-w-3xl ${
                  message.role === 'user'
                    ? 'chat-bubble-user text-sm lg:text-base text-white'
                    : 'chat-bubble-assistant text-sm lg:text-base'
                }`}
              >
                <div className="prose prose-sm max-w-none">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeKatex]}>
                    {message.content}
                  </ReactMarkdown>
                </div>

                {message.sources && message.sources.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <p className="text-xs text-gray-600 font-medium mb-2">
                      📚 Sources ({message.sources.length}):
                    </p>
                    <div className="space-y-2">
                      {message.sources.slice(0, 3).map((source, idx) => (
                        <div key={idx} className="text-xs bg-gray-50 p-2 border border-gray-200">
                          <div className="flex items-center justify-between mb-1">
                            <div className="font-medium text-black truncate flex-1">
                              {source.filename}
                            </div>
                            {source.document_id && (
                              <button
                                onClick={() => setPreviewDocument({
                                  id: source.document_id,
                                  filename: source.filename
                                })}
                                className="ml-2 px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 text-black border border-gray-300 transition-colors flex items-center space-x-1"
                                title="Preview document"
                              >
                                <EyeIcon className="h-3 w-3" />
                                <span>View</span>
                              </button>
                            )}
                          </div>
                          <div className="text-gray-600 leading-relaxed">
                            {source.chunk_text?.substring(0, 120)}...
                          </div>
                          {source.similarity && (
                            <div className="mt-1 text-xs text-gray-400">
                              Relevance: {Math.round(source.similarity * 100)}%
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="mt-2 text-xs text-gray-400">
                  {message.timestamp?.toLocaleTimeString()}
                </div>
              </div>

              {message.role === 'user' && (
                <div className="flex-shrink-0 w-8 h-8 lg:w-10 lg:h-10 bg-gray-200 border border-gray-300 flex items-center justify-center">
                  <UserIcon className="w-4 h-4 lg:w-5 lg:h-5 text-black" />
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
            <div className="flex-shrink-0 w-8 h-8 lg:w-10 lg:h-10 bg-black border border-black flex items-center justify-center">
              <SparklesIcon className="w-4 h-4 lg:w-5 lg:h-5 text-white animate-pulse" />
            </div>
            <div className="chat-bubble-assistant text-sm lg:text-base">
              <div className="loading-dots text-black">
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
        <div className="border-t border-gray-200 bg-white p-3 lg:p-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex items-end space-x-2 lg:space-x-3">
              <button
                onClick={handleAttachClick}
                className="p-3 lg:p-4 border border-gray-300 hover:bg-gray-50"
                title="Attach files"
              >
                <PaperClipIcon className="w-4 h-4 lg:w-5 lg:h-5" />
                <input ref={fileInputRef} type="file" onChange={handleFilesSelected} className="hidden" multiple accept="image/*,application/pdf,text/plain,application/vnd.openxmlformats-officedocument.wordprocessingml.document" />
              </button>
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
            
            <div className="mt-2 text-xs text-gray-500 text-center hidden lg:block">
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
