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
import ChatInterface from '@/components/chat/chat-interface'
import Sidebar from '@/components/layout/sidebar'
import StatsCards from '@/components/dashboard/stats-cards'
import WelcomeScreen from '@/components/welcome/welcome-screen'
import AnalyticsDashboard from '@/components/analytics/analytics-dashboard'
import DocumentUpload from '@/components/upload/document-upload'
import Dashboard from '@/components/dashboard/dashboard'
import DocumentsList from '@/components/documents/documents-list'
import ContractsList from '@/components/contracts/contracts-list'
import ChunksList from '@/components/chunks/chunks-list'
import Settings from '@/components/settings/settings'
import { useSystemStats } from '@/hooks/use-api'
import toast from 'react-hot-toast'

type View = 'dashboard' | 'chat' | 'analytics' | 'upload' | 'settings' | 'documents' | 'contracts' | 'chunks'

export default function HomePage() {
  const [currentView, setCurrentView] = useState<View>('dashboard')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [hasStartedChat, setHasStartedChat] = useState(false)
  const [showSearchModal, setShowSearchModal] = useState(false)
  const { data: stats, isLoading: statsLoading } = useSystemStats()

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
    switch (currentView) {
      case 'dashboard':
        return stats ? <Dashboard stats={stats} onNavigate={setCurrentView} /> : <div className="p-8 text-center">Loading...</div>
      case 'chat':
        if (!hasStartedChat) {
          return (
            <WelcomeScreen 
              onStartChat={handleStartChat}
              onSampleQuestion={handleSampleQuestion}
              sampleQuestions={sampleQuestions}
            />
          )
        }
        return <ChatInterface />
      case 'documents':
        return <DocumentsList />
      case 'contracts':
        return <ContractsList />
      case 'chunks':
        return <ChunksList />
      case 'analytics':
        return <AnalyticsDashboard />
      case 'upload':
        return <DocumentUpload />
      case 'settings':
        return <Settings stats={stats} />
      default:
        return stats ? <Dashboard stats={stats} onNavigate={setCurrentView} /> : <div className="p-8 text-center">Loading...</div>
    }
  }

  return (
    <div className="flex h-screen bg-white">

      {/* Main Content Area - Full Width */}
      <div className="w-full flex flex-col min-w-0">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 px-4 lg:px-6 py-3 lg:py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 lg:space-x-4">
              
            <button 
              onClick={() => setCurrentView('dashboard')}
              className="flex items-center space-x-2 lg:space-x-3 hover:opacity-80 transition-opacity"
            >
                <div className="p-1.5 lg:p-2 bg-black border border-black">
                  <ChatBubbleLeftRightIcon className="h-5 w-5 lg:h-6 lg:w-6 text-white" />
                </div>
                <div className="hidden sm:block">
                  <h1 className="text-lg lg:text-xl font-semibold text-black">
                    CLM Automation
                  </h1>
                  <p className="text-xs lg:text-sm text-gray-600">
                    AI-powered contract management
                  </p>
                </div>
              </button>
            </div>

            <div className="flex items-center space-x-2 lg:space-x-4">
              {/* System Status - Hidden on mobile */}
              <div className="hidden md:flex items-center space-x-2 text-xs lg:text-sm">
                <div className="w-2 h-2 bg-black rounded-full animate-pulse"></div>
                <span className="text-gray-600">Online</span>
              </div>

              {/* View Toggle Buttons - Responsive */}
              <div className="flex items-center bg-gray-100 p-1 border border-gray-200">
                {[
                  { id: 'dashboard', icon: SparklesIcon, label: 'Dashboard' },
                  { id: 'chat', icon: ChatBubbleLeftRightIcon, label: 'Chat' },
                  { id: 'analytics', icon: ChartBarIcon, label: 'Analytics' },
                  { id: 'upload', icon: DocumentArrowUpIcon, label: 'Upload' },
                  { id: 'settings', icon: CogIcon, label: 'Settings' },
                ].map((view) => {
                  const IconComponent = view.icon
                  return (
                    <motion.button
                      key={view.id}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => setCurrentView(view.id as View)}
                      className={`p-1.5 lg:p-2 border transition-colors ${
                        currentView === view.id
                          ? 'bg-black text-white border-black'
                          : 'bg-white text-black border-gray-300 hover:border-gray-400 hover:bg-gray-50'
                      }`}
                      title={view.label}
                    >
                      <IconComponent className="h-4 w-4 lg:h-5 lg:w-5" />
                    </motion.button>
                  )
                })}
              </div>

              {/* Global Search Button */}
              <button
                onClick={() => setShowSearchModal(true)}
                className="px-3 py-2 bg-black text-white text-sm border border-black hover:opacity-90"
              >
                Global Search
              </button>
            </div>
          </div>
        </header>


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
          className="fixed inset-0 bg-black bg-opacity-20 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Global Search Modal with iframe */}
      {showSearchModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/50" onClick={() => setShowSearchModal(false)} />
          <div className="relative bg-white w-[95%] h-[90%] max-w-6xl shadow-xl border">
            <div className="flex items-center justify-between p-3 border-b">
              <h3 className="font-medium">Global Search</h3>
              <button onClick={() => setShowSearchModal(false)} className="text-sm px-2 py-1 border">Close</button>
            </div>
            <iframe
              src="/search"
              title="Global Search"
              className="w-full h-[calc(100%-48px)]"
            />
          </div>
        </div>
      )}
    </div>
  )
}