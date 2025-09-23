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
      <div className="h-full p-8 bg-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-black mx-auto mb-4"></div>
          <p className="text-gray-600">Loading analytics...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="h-full p-8 bg-white flex items-center justify-center">
        <div className="text-center">
          <ExclamationTriangleIcon className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <p className="text-red-600">Failed to load analytics data</p>
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
    <div className="h-full overflow-auto bg-white">
      <div className="max-w-6xl mx-auto p-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-black mb-2">
            Contract Analytics Dashboard
          </h1>
          <p className="text-gray-600">
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
                <p className="text-sm text-gray-600">Total Contracts</p>
                <p className="text-2xl font-bold text-black">{totalContracts}</p>
              </div>
              <DocumentTextIcon className="h-8 w-8 text-black" />
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
                <p className="text-sm text-gray-600">Expiring Soon</p>
                <p className="text-2xl font-bold text-red-600">{data.expiring_contracts}</p>
              </div>
              <ClockIcon className="h-8 w-8 text-red-500" />
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
                <p className="text-sm text-gray-600">Departments</p>
                <p className="text-2xl font-bold text-black">{departmentEntries.length}</p>
              </div>
              <ChartBarIcon className="h-8 w-8 text-black" />
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
                <p className="text-sm text-gray-600">Total Value</p>
                <p className="text-2xl font-bold text-black">${data.total_value.toLocaleString()}</p>
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
            <h2 className="text-xl font-semibold text-black mb-4">
              Contracts by Department
            </h2>
            {departmentEntries.length > 0 ? (
              <div className="space-y-4">
                {departmentEntries.map(([department, count]) => {
                  const percentage = totalContracts > 0 ? (count / totalContracts * 100).toFixed(1) : 0
                  return (
                    <div key={department} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium text-black">{department || 'Unknown'}</span>
                        <span className="text-sm text-gray-600">{count} ({percentage}%)</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-black h-2 rounded-full transition-all duration-300"
                          style={{ width: `${percentage}%` }}
                        ></div>
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <p className="text-gray-500 text-center py-8">No department data available</p>
            )}
          </motion.div>

          {/* Contract Timeline */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="card p-6"
          >
            <h2 className="text-xl font-semibold text-black mb-4">
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
                      <div key={index} className={`p-3 border ${
                        isExpiring ? 'bg-red-50 border-red-200' : 'bg-gray-50 border-gray-200'
                      }`}>
                        <div className="flex justify-between items-start">
                          <div className="flex-1">
                            <p className="font-medium text-black text-sm">{contract.name}</p>
                            <p className="text-xs text-gray-600">{contract.department}</p>
                          </div>
                          <div className="text-right">
                            <p className={`text-xs font-medium ${
                              isExpiring ? 'text-red-700' : 'text-black'
                            }`}>
                              {daysUntilExpiry > 0 ? `${daysUntilExpiry} days` : 'Expired'}
                            </p>
                            <p className="text-xs text-gray-500">
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
              <p className="text-gray-500 text-center py-8">No contract timeline data available</p>
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
          <h2 className="text-xl font-semibold text-black mb-4">
            System Status
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
            <div className="p-4 bg-gray-100 border border-gray-200">
              <div className="text-2xl mb-2">💰</div>
              <p className="text-sm font-medium text-black">Total Value</p>
              <p className="text-lg text-gray-800">${data.total_value.toLocaleString()}</p>
            </div>
            <div className="p-4 bg-gray-100 border border-gray-200">
              <div className="text-2xl mb-2">🤖</div>
              <p className="text-sm font-medium text-black">AI Processing Active</p>
              <p className="text-xs text-gray-600">Contract analysis ready</p>
            </div>
            <div className="p-4 bg-gray-100 border border-gray-200">
              <div className="text-2xl mb-2">📊</div>
              <p className="text-sm font-medium text-black">Data Synchronized</p>
              <p className="text-xs text-gray-600">Last updated: just now</p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
