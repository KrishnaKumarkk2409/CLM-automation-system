'use client'

import { motion } from 'framer-motion'
import { useAnalytics } from '@/hooks/use-api'
import { 
  ChartBarIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline'

export default function AnalyticsDashboard() {
  const { data: analytics, isLoading, error } = useAnalytics()

  if (isLoading) {
    return (
      <div className="h-full p-8 bg-gradient-to-br from-primary-50 via-white to-accent-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500 mx-auto mb-4"></div>
          <p className="text-secondary-600">Loading analytics...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="h-full p-8 bg-gradient-to-br from-primary-50 via-white to-accent-50 flex items-center justify-center">
        <div className="text-center">
          <ExclamationTriangleIcon className="h-12 w-12 text-danger-500 mx-auto mb-4" />
          <p className="text-danger-600">Failed to load analytics data</p>
        </div>
      </div>
    )
  }

  const data = analytics || {
    contract_timeline: [],
    department_distribution: {},
    expiring_contracts: 0,
    total_value: 0
  }

  const departmentEntries = Object.entries(data.department_distribution)
  const totalContracts = departmentEntries.reduce((sum, [_, count]) => sum + count, 0)

  return (
    <div className="h-full overflow-auto bg-gradient-to-br from-primary-50 via-white to-accent-50">
      <div className="max-w-6xl mx-auto p-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-secondary-900 mb-2">
            Contract Analytics Dashboard
          </h1>
          <p className="text-secondary-600">
            Overview of your contract portfolio and key metrics
          </p>
        </motion.div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="card p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-secondary-600">Total Contracts</p>
                <p className="text-2xl font-bold text-secondary-900">{totalContracts}</p>
              </div>
              <DocumentTextIcon className="h-8 w-8 text-primary-500" />
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="card p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-secondary-600">Expiring Soon</p>
                <p className="text-2xl font-bold text-warning-600">{data.expiring_contracts}</p>
              </div>
              <ClockIcon className="h-8 w-8 text-warning-500" />
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="card p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-secondary-600">Departments</p>
                <p className="text-2xl font-bold text-accent-600">{departmentEntries.length}</p>
              </div>
              <ChartBarIcon className="h-8 w-8 text-accent-500" />
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="card p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-secondary-600">Total Value</p>
                <p className="text-2xl font-bold text-success-600">${data.total_value.toLocaleString()}</p>
              </div>
              <div className="text-2xl">💰</div>
            </div>
          </motion.div>
        </div>

        {/* Department Distribution */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="card p-6"
          >
            <h2 className="text-xl font-semibold text-secondary-900 mb-4">
              Contracts by Department
            </h2>
            {departmentEntries.length > 0 ? (
              <div className="space-y-4">
                {departmentEntries.map(([department, count]) => {
                  const percentage = totalContracts > 0 ? (count / totalContracts * 100).toFixed(1) : 0
                  return (
                    <div key={department} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium text-secondary-700">{department || 'Unknown'}</span>
                        <span className="text-sm text-secondary-600">{count} ({percentage}%)</span>
                      </div>
                      <div className="w-full bg-secondary-200 rounded-full h-2">
                        <div 
                          className="bg-primary-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${percentage}%` }}
                        ></div>
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <p className="text-secondary-500 text-center py-8">No department data available</p>
            )}
          </motion.div>

          {/* Contract Timeline */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="card p-6"
          >
            <h2 className="text-xl font-semibold text-secondary-900 mb-4">
              Upcoming Expirations
            </h2>
            {data.contract_timeline.length > 0 ? (
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {data.contract_timeline
                  .filter(contract => contract.end_date)
                  .sort((a, b) => new Date(a.end_date).getTime() - new Date(b.end_date).getTime())
                  .slice(0, 8)
                  .map((contract, index) => {
                    const endDate = new Date(contract.end_date)
                    const today = new Date()
                    const daysUntilExpiry = Math.ceil((endDate.getTime() - today.getTime()) / (1000 * 60 * 60 * 24))
                    const isExpiring = daysUntilExpiry <= 30
                    
                    return (
                      <div key={index} className={`p-3 rounded-lg border ${
                        isExpiring ? 'bg-warning-50 border-warning-200' : 'bg-secondary-50 border-secondary-200'
                      }`}>
                        <div className="flex justify-between items-start">
                          <div className="flex-1">
                            <p className="font-medium text-secondary-900 text-sm">{contract.name}</p>
                            <p className="text-xs text-secondary-600">{contract.department}</p>
                          </div>
                          <div className="text-right">
                            <p className={`text-xs font-medium ${
                              isExpiring ? 'text-warning-700' : 'text-secondary-700'
                            }`}>
                              {daysUntilExpiry > 0 ? `${daysUntilExpiry} days` : 'Expired'}
                            </p>
                            <p className="text-xs text-secondary-500">
                              {endDate.toLocaleDateString()}
                            </p>
                          </div>
                        </div>
                      </div>
                    )
                  })
                }
              </div>
            ) : (
              <p className="text-secondary-500 text-center py-8">No contract timeline data available</p>
            )}
          </motion.div>
        </div>

        {/* Status Summary */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="card p-6"
        >
          <h2 className="text-xl font-semibold text-secondary-900 mb-4">
            System Status
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
            <div className="p-4 bg-success-50 rounded-lg">
              <div className="text-2xl mb-2">✅</div>
              <p className="text-sm font-medium text-success-700">System Operational</p>
              <p className="text-xs text-success-600">All services running</p>
            </div>
            <div className="p-4 bg-primary-50 rounded-lg">
              <div className="text-2xl mb-2">🤖</div>
              <p className="text-sm font-medium text-primary-700">AI Processing Active</p>
              <p className="text-xs text-primary-600">Contract analysis ready</p>
            </div>
            <div className="p-4 bg-accent-50 rounded-lg">
              <div className="text-2xl mb-2">📊</div>
              <p className="text-sm font-medium text-accent-700">Data Synchronized</p>
              <p className="text-xs text-accent-600">Last updated: just now</p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
