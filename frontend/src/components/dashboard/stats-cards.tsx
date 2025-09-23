'use client'

import { motion } from 'framer-motion'
import { 
  DocumentTextIcon,
  DocumentCheckIcon,
  CubeIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline'
import { SystemStats } from '@/hooks/use-api'

interface StatsCardsProps {
  stats: SystemStats
}

export default function StatsCards({ stats }: StatsCardsProps) {
  const statItems = [
    {
      label: 'Documents',
      value: stats.total_documents,
      icon: DocumentTextIcon,
      color: 'primary',
      description: 'Total processed'
    },
    {
      label: 'Active Contracts',
      value: stats.active_contracts,
      icon: DocumentCheckIcon,
      color: 'success',
      description: 'Currently active'
    },
    {
      label: 'Text Chunks',
      value: stats.total_chunks.toLocaleString(),
      icon: CubeIcon,
      color: 'accent',
      description: 'Searchable segments'
    },
    {
      label: 'System Status',
      value: stats.system_status,
      icon: CheckCircleIcon,
      color: 'success',
      description: 'All systems operational'
    }
  ]

  const getColorClasses = (color: string) => {
    const colorMap = {
      primary: {
        bg: 'from-primary to-primary-500',
        text: 'text-primary',
        light: 'bg-primary-50'
      },
      success: {
        bg: 'from-success to-success-500',
        text: 'text-success',
        light: 'bg-success/10'
      },
      warning: {
        bg: 'from-warning to-warning-500',
        text: 'text-warning',
        light: 'bg-warning/10'
      },
      destructive: {
        bg: 'from-destructive to-destructive-500',
        text: 'text-destructive',
        light: 'bg-destructive/10'
      }
    }
    return colorMap[color as keyof typeof colorMap] || colorMap.primary
  }

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 lg:gap-4">
      {statItems.map((stat, index) => {
        const IconComponent = stat.icon
        const colors = getColorClasses(stat.color)
        
        return (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
            className="card-hover"
          >
            <div className="p-3 lg:p-4">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-2 lg:space-x-3">
                    <div className={`p-1.5 lg:p-2 rounded-lg ${colors.light}`}>
                      <IconComponent className={`h-4 w-4 lg:h-5 lg:w-5 ${colors.text}`} />
                    </div>
                    <div className="min-w-0">
                      <p className="text-lg lg:text-2xl font-semibold text-secondary-900 truncate">
                        {typeof stat.value === 'string' ? 
                          stat.value.charAt(0).toUpperCase() + stat.value.slice(1) : 
                          stat.value
                        }
                      </p>
                      <p className="text-xs lg:text-sm font-medium text-secondary-600 truncate">
                        {stat.label}
                      </p>
                      <p className="text-xs text-secondary-500 hidden lg:block">
                        {stat.description}
                      </p>
                    </div>
                  </div>
                </div>
                
                {/* Optional trend indicator */}
                {stat.label !== 'System Status' && (
                  <div className="flex items-center space-x-1">
                    <motion.div
                      className="w-2 h-2 bg-success-500 rounded-full"
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    />
                    <span className="text-xs text-success-600 font-medium">
                      Active
                    </span>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )
      })}
    </div>
  )
}