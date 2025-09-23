'use client'

import { motion } from 'framer-motion'
import { 
  SparklesIcon, 
  ChatBubbleLeftRightIcon,
  DocumentTextIcon,
  ShieldCheckIcon,
  ChartBarIcon,
  ClockIcon,
  BoltIcon
} from '@heroicons/react/24/outline'
import { 
  PlayIcon 
} from '@heroicons/react/24/solid'

interface WelcomeScreenProps {
  onStartChat: () => void
  onSampleQuestion: (question: string) => void
  sampleQuestions: string[]
}

export default function WelcomeScreen({ 
  onStartChat, 
  onSampleQuestion, 
  sampleQuestions 
}: WelcomeScreenProps) {
  const features = [
    {
      icon: ChatBubbleLeftRightIcon,
      title: "AI-Powered Analysis",
      description: "Get instant insights from your contract documents with advanced natural language processing"
    },
    {
      icon: ShieldCheckIcon,
      title: "Risk Management",
      description: "Identify potential conflicts and compliance issues before they become problems"
    },
    {
      icon: ChartBarIcon,
      title: "Smart Analytics",
      description: "Visualize contract timelines, department distributions, and key performance metrics"
    },
    {
      icon: ClockIcon,
      title: "Expiration Tracking",
      description: "Never miss a renewal deadline with intelligent contract monitoring and alerts"
    }
  ]

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2
      }
    }
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5, ease: "easeOut" }
    }
  }

  return (
    <div className="h-full overflow-auto bg-gradient-to-br from-primary-50 via-white to-accent-50">
      <motion.div 
        className="max-w-6xl mx-auto px-3 sm:px-4 lg:px-6 py-4 sm:py-6 lg:py-12"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Hero Section */}
        <motion.div className="text-center mb-6 sm:mb-8 lg:mb-12" variants={itemVariants}>
          <motion.div 
            className="inline-flex items-center justify-center w-14 h-14 sm:w-16 sm:h-16 lg:w-20 lg:h-20 bg-gradient-to-br from-primary-500 to-accent-500 rounded-full mb-3 sm:mb-4 lg:mb-6 shadow-lg"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <SparklesIcon className="w-6 h-6 sm:w-8 sm:h-8 lg:w-10 lg:h-10 text-white" />
          </motion.div>
          
          <h1 className="text-2xl sm:text-3xl lg:text-5xl font-bold text-secondary-900 mb-2 sm:mb-3 lg:mb-4">
            Welcome to{' '}
            <span className="gradient-text">CLM Automation</span>
          </h1>
          
          <p className="text-sm sm:text-base lg:text-xl text-secondary-600 mb-4 sm:mb-6 lg:mb-8 max-w-2xl mx-auto px-2 sm:px-4">
            Transform your contract management with AI-powered insights, automated analysis, and intelligent monitoring
          </p>

          <motion.button
            onClick={onStartChat}
            className="btn-primary text-sm sm:text-base lg:text-lg px-4 sm:px-6 lg:px-8 py-2 sm:py-3 lg:py-4 rounded-xl shadow-lg"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <PlayIcon className="w-5 h-5 lg:w-6 lg:h-6 mr-2" />
            Start Chatting
          </motion.button>
        </motion.div>

        {/* Features Grid */}
        <motion.div 
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 mb-8 sm:mb-12"
          variants={itemVariants}
        >
          {features.map((feature, index) => {
            const IconComponent = feature.icon
            return (
              <motion.div
                key={index}
                className="card-hover p-4 sm:p-6 text-center"
                whileHover={{ scale: 1.02 }}
                variants={itemVariants}
              >
                <div className="w-10 h-10 sm:w-12 sm:h-12 bg-gradient-to-br from-primary-100 to-accent-100 rounded-lg flex items-center justify-center mx-auto mb-3 sm:mb-4">
                  <IconComponent className="w-5 h-5 sm:w-6 sm:h-6 text-primary-600" />
                </div>
                <h3 className="font-semibold text-sm sm:text-base text-secondary-900 mb-1 sm:mb-2">
                  {feature.title}
                </h3>
                <p className="text-xs sm:text-sm text-secondary-600">
                  {feature.description}
                </p>
              </motion.div>
            )
          })}
        </motion.div>

        {/* Sample Questions */}
        <motion.div variants={itemVariants}>
          <h2 className="text-xl sm:text-2xl font-semibold text-secondary-900 text-center mb-6 sm:mb-8">
            💭 Try asking me something like...
          </h2>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4 max-w-4xl mx-auto">
            {sampleQuestions.map((question, index) => (
              <motion.button
                key={index}
                onClick={() => onSampleQuestion(question)}
                className="card-hover p-3 sm:p-4 text-left hover:bg-gradient-to-r hover:from-primary-50 hover:to-accent-50 transition-all duration-300"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                variants={itemVariants}
              >
                <div className="flex items-start space-x-2 sm:space-x-3">
                  <div className="flex-shrink-0 w-6 h-6 sm:w-8 sm:h-8 bg-gradient-to-br from-primary-100 to-accent-100 rounded-lg flex items-center justify-center mt-0.5 sm:mt-1">
                    <BoltIcon className="w-3 h-3 sm:w-4 sm:h-4 text-primary-600" />
                  </div>
                  <div>
                    <p className="text-xs sm:text-sm text-secondary-800 font-medium">
                      {question}
                    </p>
                  </div>
                </div>
              </motion.button>
            ))}
          </div>
        </motion.div>

        {/* Quick Stats Teaser */}
        <motion.div 
          className="mt-10 sm:mt-16 text-center"
          variants={itemVariants}
        >
          <div className="bg-gradient-to-r from-primary-600 to-accent-600 rounded-xl sm:rounded-2xl p-5 sm:p-8 text-white shadow-xl">
            <h3 className="text-xl sm:text-2xl font-semibold mb-3 sm:mb-4">
              Ready to revolutionize your contract management?
            </h3>
            <div className="grid grid-cols-3 divide-x divide-white/20 max-w-md mx-auto">
              <div className="px-2 sm:px-4">
                <div className="text-2xl sm:text-3xl font-bold">AI</div>
                <div className="text-xs sm:text-sm opacity-90">Powered</div>
              </div>
              <div className="px-2 sm:px-4">
                <div className="text-2xl sm:text-3xl font-bold">24/7</div>
                <div className="text-xs sm:text-sm opacity-90">Available</div>
              </div>
              <div className="px-2 sm:px-4">
                <div className="text-2xl sm:text-3xl font-bold">∞</div>
                <div className="text-xs sm:text-sm opacity-90">Contracts</div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* System Status */}
        <motion.div 
          className="mt-6 sm:mt-8 text-center"
          variants={itemVariants}
        >
          <div className="inline-flex items-center space-x-1 sm:space-x-2 text-xs sm:text-sm text-secondary-600 bg-white/80 backdrop-blur-sm rounded-lg px-3 sm:px-4 py-1.5 sm:py-2 border border-secondary-200/50">
            <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-success-500 rounded-full animate-pulse"></div>
            <span>System online and ready</span>
          </div>
        </motion.div>
      </motion.div>
    </div>
  )
}