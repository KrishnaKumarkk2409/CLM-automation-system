'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  PaperAirplaneIcon, 
  DocumentTextIcon,
  ChartBarIcon,
  CogIcon,
  Bars3Icon,
  XMarkIcon,
  PlusIcon,
  ClockIcon,
  DocumentArrowUpIcon
} from '@heroicons/react/24/outline'
import { 
  ChatBubbleLeftRightIcon,
  SparklesIcon,
  ShieldCheckIcon
} from '@heroicons/react/24/solid'
import Header from '@/components/layout/Header'
import ChatInterface from '@/components/chat/chat-interface'
import Sidebar from '@/components/layout/sidebar'
import StatsCards from '@/components/dashboard/stats-cards'
import WelcomeScreen from '@/components/welcome/welcome-screen'
import AnalyticsDashboard from '@/components/analytics/enhanced-analytics-dashboard'
import CleanDashboard from '@/components/dashboard/clean-dashboard'
import DocumentUpload from '@/components/upload/document-upload'
import SettingsPage from '@/components/settings/settings-page'
import { useSystemStats } from '@/hooks/use-api'
import toast from 'react-hot-toast'
import { useAdvancedDocumentSearch } from '@/hooks/use-api'
import FloatingChatButton from '@/components/FloatingChatButton'
import ChatModal from '@/components/chat/ChatModal'

type View = 'dashboard' | 'chat' | 'analytics' | 'upload' | 'settings'

export default function HomePage() {
  const [currentView, setCurrentView] = useState<View>('dashboard')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [hasStartedChat, setHasStartedChat] = useState(false)
  const { data: stats, isLoading: statsLoading } = useSystemStats()
  const [searchTerm, setSearchTerm] = useState('')
  const [showSearchModal, setShowSearchModal] = useState(false)
  const [searchResults, setSearchResults] = useState<any[] | null>(null)
  const advancedSearch = useAdvancedDocumentSearch()
  const [chatOpen, setChatOpen] = useState(false)

  // Sample questions for welcome screen
  const sampleQuestions = [
    "📅 Show me contracts expiring in the next 30 days",
    "📊 What's the total value of all active contracts?",
    "⚠️ Are there any contract conflicts I should know about?",
    "🔍 Find all contracts with TechCorp",
    "📋 Give me a summary of our software licenses",
    "📈 Show contract analytics and department distribution",
    "🔄 What contracts need renewal soon?",
    "📝 What are the key terms in our NDA agreements?"
  ]

  const handleStartChat = () => {
    setHasStartedChat(true)
    toast.success('Ready to help! Ask me anything about your contracts.')
  }

  const handleSampleQuestion = (question: string) => {
    const cleanQuestion = question.split(' ').slice(1).join(' ') // Remove emoji
    setHasStartedChat(true)
    // This would trigger the chat with the selected question
    // Implementation depends on your chat component structure
  }

  const renderCurrentView = () => {
    if (currentView === 'chat' && !hasStartedChat) {
      return (
        <WelcomeScreen 
          onStartChat={handleStartChat}
          onSampleQuestion={handleSampleQuestion}
          sampleQuestions={sampleQuestions}
        />
      )
    }

    switch (currentView) {
      case 'dashboard':
        return (
          <CleanDashboard 
            onUpload={() => setCurrentView('upload')}
            onAnalytics={() => setCurrentView('analytics')}
            onStartChat={handleStartChat}
          />
        )
      case 'chat':
        return <ChatInterface />
      case 'analytics':
        return <AnalyticsDashboard />
      case 'upload':
        return <DocumentUpload />
      case 'settings':
        return <SettingsPage />
      default:
        return <ChatInterface />
    }
  }

  const runHeaderSearch = async () => {
    if (!searchTerm.trim()) return
    try {
      const data = await advancedSearch.mutateAsync({ query: searchTerm, limit: 12 })
      setSearchResults(data.documents || [])
      setShowSearchModal(true)
    } catch (e) {
      toast.error('Search failed')
    }
  }

  return (
    <div className="flex h-screen bg-gradient-to-br from-primary-50 via-white to-accent-50">

      {/* Main Content Area - Full Width */}
      <div className="w-full flex flex-col min-w-0">
        {/* Header - Using the new Header component */}
        <Header onSearchResults={(results) => {
          setSearchResults(results);
          setShowSearchModal(true);
        }} />

        {/* Stats Cards - Responsive */}
        {!statsLoading && stats && (
          <div className="px-4 lg:px-6 py-3 lg:py-4 bg-white/50 border-b border-secondary-200/50">
            <StatsCards stats={stats} />
          </div>
        )}

        {/* Main Content */}
        <main className="flex-1 overflow-hidden">
          <AnimatePresence mode="wait">
            <motion.div
              key={currentView}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="h-full"
            >
              {renderCurrentView()}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>

      {/* Mobile Overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Floating Chat */}
      <FloatingChatButton onClick={() => setChatOpen(true)} />
      <ChatModal open={chatOpen} onClose={() => setChatOpen(false)} />

      {/* Search Modal */}
      {showSearchModal && (
        <div className="fixed inset-0 bg-black/30 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-lg border border-secondary-200 w-full max-w-3xl">
            <div className="flex items-center justify-between px-4 py-3 border-b border-secondary-200">
              <h3 className="text-sm font-semibold text-secondary-900">Search Results</h3>
              <button onClick={() => setShowSearchModal(false)} className="text-secondary-500 hover:text-secondary-900">
                ✕
              </button>
            </div>
            <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-3 max-h-[60vh] overflow-auto">
              {(searchResults || []).map((doc, i) => (
                <div key={i} className="border border-secondary-200 rounded-lg p-3 bg-secondary-50">
                  <div className="text-sm font-medium text-secondary-900 truncate">{doc.filename}</div>
                  <div className="text-xs text-secondary-600 mt-1">{doc.file_type?.toUpperCase()} • {Math.round((doc.similarity || 0) * 100)}%</div>
                  <div className="text-xs text-secondary-700 italic mt-2">{doc.relevant_excerpt?.slice(0, 160)}...</div>
                </div>
              ))}
              {(!searchResults || searchResults.length === 0) && (
                <div className="text-center text-secondary-500 py-12">No results</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}