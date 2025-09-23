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
    <div className="h-full overflow-auto bg-white">
      <motion.div 
        className="max-w-6xl mx-auto px-4 lg:px-6 py-6 lg:py-12"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Hero Section */}
        <motion.div className="text-center mb-8 lg:mb-12" variants={itemVariants}>
          <motion.div 
            className="inline-flex items-center justify-center w-16 h-16 lg:w-20 lg:h-20 bg-black border border-black mb-4 lg:mb-6"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <SparklesIcon className="w-8 h-8 lg:w-10 lg:h-10 text-white" />
          </motion.div>
          
          <h1 className="text-3xl lg:text-5xl font-bold text-black mb-3 lg:mb-4">
            Welcome to{' '}
            <span className="gradient-text">CLM Automation</span>
          </h1>
          
          <p className="text-base lg:text-xl text-gray-600 mb-6 lg:mb-8 max-w-2xl mx-auto px-4">
            Transform your contract management with AI-powered insights, automated analysis, and intelligent monitoring
          </p>

          <motion.button
            onClick={onStartChat}
            className="btn-primary text-base lg:text-lg px-6 lg:px-8 py-3 lg:py-4"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <PlayIcon className="w-5 h-5 lg:w-6 lg:h-6 mr-2" />
            Start Chatting
          </motion.button>
        </motion.div>

        {/* Features Grid */}
        <motion.div 
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12"
          variants={itemVariants}
        >
          {features.map((feature, index) => {
            const IconComponent = feature.icon
            return (
              <motion.div
                key={index}
                className="card-hover p-6 text-center"
                whileHover={{ scale: 1.02 }}
                variants={itemVariants}
              >
                <div className="w-12 h-12 bg-gray-100 border border-gray-200 flex items-center justify-center mx-auto mb-4">
                  <IconComponent className="w-6 h-6 text-black" />
                </div>
                <h3 className="font-semibold text-black mb-2">
                  {feature.title}
                </h3>
                <p className="text-sm text-gray-600">
                  {feature.description}
                </p>
              </motion.div>
            )
          })}
        </motion.div>

        {/* Sample Questions */}
        <motion.div variants={itemVariants}>
          <h2 className="text-2xl font-semibold text-black text-center mb-8">
            💭 Try asking me something like...
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-4xl mx-auto">
            {sampleQuestions.map((question, index) => (
              <motion.button
                key={index}
                onClick={() => onSampleQuestion(question)}
                className="card-hover p-4 text-left hover:bg-gray-50 transition-all duration-300"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                variants={itemVariants}
              >
                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0 w-8 h-8 bg-gray-100 border border-gray-200 flex items-center justify-center mt-1">
                    <BoltIcon className="w-4 h-4 text-black" />
                  </div>
                  <div>
                    <p className="text-black font-medium">
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
          className="mt-16 text-center"
          variants={itemVariants}
        >
          <div className="bg-black border border-black p-8 text-white">
            <h3 className="text-2xl font-semibold mb-4">
              Ready to revolutionize your contract management?
            </h3>
            <div className="grid grid-cols-3 divide-x divide-white max-w-md mx-auto">
              <div className="px-4">
                <div className="text-3xl font-bold">AI</div>
                <div className="text-sm">Powered</div>
              </div>
              <div className="px-4">
                <div className="text-3xl font-bold">24/7</div>
                <div className="text-sm">Available</div>
              </div>
              <div className="px-4">
                <div className="text-3xl font-bold">∞</div>
                <div className="text-sm">Contracts</div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* System Status */}
        <motion.div 
          className="mt-8 text-center"
          variants={itemVariants}
        >
          <div className="inline-flex items-center space-x-2 text-sm text-gray-600 bg-white px-4 py-2 border border-gray-200">
            <div className="w-2 h-2 bg-black rounded-full animate-pulse"></div>
            <span>System online and ready</span>
          </div>
        </motion.div>
      </motion.div>
    </div>
  )
}