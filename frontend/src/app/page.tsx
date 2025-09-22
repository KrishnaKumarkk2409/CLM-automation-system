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
import { useSystemStats } from '@/hooks/use-api'
import toast from 'react-hot-toast'

type View = 'chat' | 'analytics' | 'upload' | 'settings'

export default function HomePage() {
  const [currentView, setCurrentView] = useState<View>('chat')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [hasStartedChat, setHasStartedChat] = useState(false)
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
      case 'chat':
        return <ChatInterface />
      case 'analytics':
        return <AnalyticsDashboard />
      case 'upload':
        return <DocumentUpload />
      case 'settings':
        return <div className="p-8"><div className="text-center">Settings coming soon...</div></div>
      default:
        return <ChatInterface />
    }
  }

  return (
    <div className="flex h-screen bg-gradient-to-br from-primary-50 via-white to-accent-50">

      {/* Main Content Area - Full Width */}
      <div className="w-full flex flex-col min-w-0">
        {/* Header */}
        <header className="bg-white/80 backdrop-blur-sm border-b border-secondary-200/50 px-4 lg:px-6 py-3 lg:py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 lg:space-x-4">
              
              <div className="flex items-center space-x-2 lg:space-x-3">
                <div className="p-1.5 lg:p-2 bg-gradient-to-br from-primary-500 to-accent-500 rounded-lg shadow-sm">
                  <ChatBubbleLeftRightIcon className="h-5 w-5 lg:h-6 lg:w-6 text-white" />
                </div>
                <div className="hidden sm:block">
                  <h1 className="text-lg lg:text-xl font-semibold text-secondary-900">
                    CLM Automation
                  </h1>
                  <p className="text-xs lg:text-sm text-secondary-500">
                    AI-powered contract management
                  </p>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-2 lg:space-x-4">
              {/* System Status - Hidden on mobile */}
              <div className="hidden md:flex items-center space-x-2 text-xs lg:text-sm">
                <div className="w-2 h-2 bg-success-500 rounded-full animate-pulse"></div>
                <span className="text-secondary-600">Online</span>
              </div>

              {/* View Toggle Buttons - Responsive */}
              <div className="flex items-center bg-secondary-100 p-1 rounded-lg">
                {[
                  { id: 'chat', icon: ChatBubbleLeftRightIcon, label: 'Chat' },
                  { id: 'analytics', icon: ChartBarIcon, label: 'Analytics' },
                  { id: 'upload', icon: DocumentArrowUpIcon, label: 'Upload' },
                ].map((view) => {
                  const IconComponent = view.icon
                  return (
                    <motion.button
                      key={view.id}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => setCurrentView(view.id as View)}
                      className={`p-1.5 lg:p-2 rounded-md transition-colors ${
                        currentView === view.id
                          ? 'bg-white shadow-sm text-primary-600'
                          : 'text-secondary-600 hover:text-secondary-900'
                      }`}
                      title={view.label}
                    >
                      <IconComponent className="h-4 w-4 lg:h-5 lg:w-5" />
                    </motion.button>
                  )
                })}
              </div>
            </div>
          </div>
        </header>

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
    </div>
  )
}