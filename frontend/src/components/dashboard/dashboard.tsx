'use client'

import { motion } from 'framer-motion'
import { 
  DocumentTextIcon,
  DocumentCheckIcon,
  CubeIcon,
  CheckCircleIcon,
  ChartBarIcon,
  PlusIcon
} from '@heroicons/react/24/outline'
import { SystemStats } from '@/hooks/use-api'

interface DashboardProps {
  stats: SystemStats
  onNavigate: (view: 'dashboard' | 'chat' | 'analytics' | 'upload' | 'settings' | 'documents' | 'contracts' | 'chunks') => void
}

export default function Dashboard({ stats, onNavigate }: DashboardProps) {
  const dashboardCards = [
    {
      title: 'Documents',
      value: stats.total_documents,
      description: 'Total processed',
      icon: DocumentTextIcon,
      color: 'bg-gray-100',
      clickable: true,
      onClick: () => onNavigate('documents')
    },
    {
      title: 'Active Contracts',
      value: stats.active_contracts,
      description: 'Currently active',
      icon: DocumentCheckIcon,
      color: 'bg-gray-100',
      clickable: true,
      onClick: () => onNavigate('contracts')
    },
    {
      title: 'Text Chunks',
      value: stats.total_chunks.toLocaleString(),
      description: 'Searchable segments',
      icon: CubeIcon,
      color: 'bg-gray-100',
      clickable: true,
      onClick: () => onNavigate('chunks')
    },
    {
      title: 'System Status',
      value: stats.system_status,
      description: 'All systems operational',
      icon: CheckCircleIcon,
      color: 'bg-gray-100',
      clickable: false
    }
  ]

  const actionCards = [
    {
      title: 'Start Conversation',
      description: 'Ask questions about your contracts',
      icon: ChartBarIcon,
      color: 'bg-black',
      textColor: 'text-white',
      onClick: () => onNavigate('chat')
    },
    {
      title: 'Upload Documents',
      description: 'Add new contracts and documents',
      icon: PlusIcon,
      color: 'bg-gray-100',
      textColor: 'text-black',
      onClick: () => onNavigate('upload')
    },
    {
      title: 'View Analytics',
      description: 'Explore contract insights and trends',
      icon: ChartBarIcon,
      color: 'bg-gray-100',
      textColor: 'text-black',
      onClick: () => onNavigate('analytics')
    }
  ]

  return (
    <div className="h-full overflow-auto bg-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-black mb-2">Dashboard</h1>
          <p className="text-gray-600">Overview of your contract management system</p>
        </motion.div>

        {/* Stats Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-8"
        >
          <h2 className="text-xl font-semibold text-black mb-4">System Overview</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {dashboardCards.map((card, index) => {
              const IconComponent = card.icon
              return (
                <motion.div
                  key={card.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 + index * 0.05 }}
                  whileHover={card.clickable ? { scale: 1.02 } : {}}
                  className={`card p-6 ${card.clickable ? 'cursor-pointer hover:border-gray-400' : 'cursor-default'}`}
                  onClick={card.clickable ? card.onClick : undefined}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className={`p-2 ${card.color} border border-gray-200`}>
                      <IconComponent className="h-6 w-6 text-black" />
                    </div>
                    {card.clickable && (
                      <span className="text-xs text-gray-500">Click to view</span>
                    )}
                  </div>
                  <h3 className="text-2xl font-bold text-black mb-1">
                    {typeof card.value === 'string' ? 
                      card.value.charAt(0).toUpperCase() + card.value.slice(1) : 
                      card.value
                    }
                  </h3>
                  <p className="text-sm font-medium text-gray-600 mb-1">{card.title}</p>
                  <p className="text-xs text-gray-500">{card.description}</p>
                </motion.div>
              )
            })}
          </div>
        </motion.div>

        {/* Action Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <h2 className="text-xl font-semibold text-black mb-4">Quick Actions</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {actionCards.map((card, index) => {
              const IconComponent = card.icon
              return (
                <motion.button
                  key={card.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 + index * 0.05 }}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className={`card p-6 text-left ${card.color} ${card.textColor} hover:border-gray-400`}
                  onClick={card.onClick}
                >
                  <div className="flex items-center justify-between mb-4">
                    <IconComponent className="h-8 w-8" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">{card.title}</h3>
                  <p className="text-sm opacity-75">{card.description}</p>
                </motion.button>
              )
            })}
          </div>
        </motion.div>

        {/* Recent Activity Placeholder */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mt-8"
        >
          <h2 className="text-xl font-semibold text-black mb-4">Recent Activity</h2>
          <div className="card p-6 text-center text-gray-500">
            <p>Recent document uploads and contract activities will appear here</p>
          </div>
        </motion.div>
      </div>
    </div>
  )
}