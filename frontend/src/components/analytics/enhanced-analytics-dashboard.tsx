'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  ChartBarIcon, 
  DocumentTextIcon, 
  ClockIcon,
  ExclamationTriangleIcon,
  ArrowTrendingUpIcon,
  EnvelopeIcon,
  CalendarIcon,
  BuildingOfficeIcon,
  DocumentChartBarIcon,
  CheckCircleIcon,
  XCircleIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline'
import { 
  useAnalyticsDashboard, 
  useExpiringContracts, 
  useContractConflicts,
  useGenerateAnalyticsReport 
} from '@/hooks/use-api'
import toast from 'react-hot-toast'
import LoadingSpinner from '@/components/ui/loading-spinner'

interface AnalyticsProps {
  className?: string
}

export default function EnhancedAnalyticsDashboard({ className = '' }: AnalyticsProps) {
  const [showEmailForm, setShowEmailForm] = useState(false)
  const [reportEmail, setReportEmail] = useState('')
  const [selectedDays, setSelectedDays] = useState(30)
  
  const { data: dashboardData, isLoading: dashboardLoading, refetch: refetchDashboard } = useAnalyticsDashboard()
  const { data: expiringData, isLoading: expiringLoading } = useExpiringContracts(selectedDays)
  const { data: conflictsData, isLoading: conflictsLoading } = useContractConflicts()
  const generateReport = useGenerateAnalyticsReport()

  const handleGenerateReport = async () => {
    if (!reportEmail || !reportEmail.includes('@')) {
      toast.error('Please enter a valid email address')
      return
    }

    try {
      await generateReport.mutateAsync(reportEmail)
      toast.success('Report generated and sent successfully!')
      setShowEmailForm(false)
      setReportEmail('')
    } catch (error) {
      toast.error('Failed to generate report. Please try again.')
    }
  }

  const handleRefresh = () => {
    refetchDashboard()
    toast.success('Dashboard refreshed!')
  }

  if (dashboardLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  const stats = dashboardData?.overview || {
    total_documents: 0,
    active_contracts: 0,
    total_chunks: 0,
    system_status: 'unknown'
  }

  const departmentData = Object.entries(dashboardData?.distributions?.departments || {})
    .map(([name, count], index) => ({
      name,
      contracts: count as number,
      color: ['bg-blue-500', 'bg-green-500', 'bg-purple-500', 'bg-yellow-500', 'bg-red-500', 'bg-indigo-500'][index % 6]
    }))

  const fileTypeData = Object.entries(dashboardData?.distributions?.file_types || {})

  return (
    <div className={`p-4 lg:p-8 space-y-6 max-w-7xl mx-auto ${className}`}>
      {/* Header with Actions */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <h1 className="text-2xl lg:text-3xl font-bold text-gray-900">Contract Analytics</h1>
          <p className="text-gray-600 mt-1">Comprehensive insights into your contract portfolio</p>
        </div>
        
        <div className="flex items-center gap-3">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleRefresh}
            className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors flex items-center gap-2"
          >
            <ArrowPathIcon className="h-4 w-4" />
            Refresh
          </motion.button>
          
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setShowEmailForm(true)}
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors flex items-center gap-2"
          >
            <EnvelopeIcon className="h-4 w-4" />
            Generate Report
          </motion.button>
        </div>
      </div>

      {/* Email Form Modal */}
      <AnimatePresence>
        {showEmailForm && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-white rounded-lg p-6 w-full max-w-md"
            >
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Generate Analytics Report</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Email Address
                  </label>
                  <input
                    type="email"
                    value={reportEmail}
                    onChange={(e) => setReportEmail(e.target.value)}
                    placeholder="your-email@example.com"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  />
                </div>
                <div className="text-sm text-gray-500">
                  <p>Report will include:</p>
                  <ul className="list-disc list-inside mt-1 space-y-1">
                    <li>Expiring contracts analysis</li>
                    <li>Conflict detection results</li>
                    <li>Department distribution</li>
                    <li>System health summary</li>
                  </ul>
                </div>
                <div className="flex justify-end gap-3">
                  <button
                    onClick={() => setShowEmailForm(false)}
                    className="px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
                  >
                    Cancel
                  </button>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={handleGenerateReport}
                    disabled={generateReport.isPending}
                    className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {generateReport.isPending ? 'Sending...' : 'Send Report'}
                  </motion.button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6">
        {[
          {
            label: 'Total Documents',
            value: stats.total_documents,
            icon: DocumentTextIcon,
            color: 'text-blue-600',
            bgColor: 'bg-blue-50',
            borderColor: 'border-blue-200',
            change: fileTypeData.length > 0 ? `${fileTypeData.length} types` : ''
          },
          {
            label: 'Active Contracts',
            value: stats.active_contracts,
            icon: DocumentChartBarIcon,
            color: 'text-green-600',
            bgColor: 'bg-green-50',
            borderColor: 'border-green-200',
            change: departmentData.length > 0 ? `${departmentData.length} depts` : ''
          },
          {
            label: 'Expiring Soon',
            value: expiringData?.count || 0,
            icon: ClockIcon,
            color: 'text-orange-600',
            bgColor: 'bg-orange-50',
            borderColor: 'border-orange-200',
            change: `${selectedDays} days`
          },
          {
            label: 'Conflicts Found',
            value: conflictsData?.count || 0,
            icon: ExclamationTriangleIcon,
            color: 'text-red-600',
            bgColor: 'bg-red-50',
            borderColor: 'border-red-200',
            change: conflictsData?.count === 0 ? 'All resolved' : 'Need attention'
          }
        ].map((metric, index) => {
          const IconComponent = metric.icon
          return (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`${metric.bgColor} ${metric.borderColor} p-4 lg:p-6 rounded-lg shadow-sm border`}
            >
              <div className="flex items-center justify-between">
                <div className="min-w-0 flex-1">
                  <p className="text-sm text-gray-600 truncate">{metric.label}</p>
                  <p className="text-xl lg:text-2xl font-semibold text-gray-900">{metric.value}</p>
                  {metric.change && (
                    <p className="text-xs text-gray-500 mt-1">{metric.change}</p>
                  )}
                </div>
                <IconComponent className={`h-6 w-6 lg:h-8 lg:w-8 ${metric.color} flex-shrink-0`} />
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* Content Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Department Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="xl:col-span-2 bg-white p-6 rounded-lg shadow-sm border border-gray-200"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Contracts by Department</h3>
            <BuildingOfficeIcon className="h-5 w-5 text-gray-400" />
          </div>
          
          {departmentData.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {departmentData.map((dept) => {
                const maxContracts = Math.max(...departmentData.map(d => d.contracts))
                return (
                  <div key={dept.name} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className={`w-3 h-3 rounded-full ${dept.color} flex-shrink-0`} />
                        <span className="text-sm text-gray-700 font-medium">{dept.name}</span>
                      </div>
                      <span className="text-sm font-semibold text-gray-900">{dept.contracts}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className={`${dept.color} h-2 rounded-full transition-all duration-700 ease-out`}
                        style={{ width: `${(dept.contracts / maxContracts) * 100}%` }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="text-center text-gray-500 py-12">
              <BuildingOfficeIcon className="h-12 w-12 mx-auto mb-3 text-gray-300" />
              <p className="font-medium">No department data available</p>
              <p className="text-sm">Upload contracts to see department distribution</p>
            </div>
          )}
        </motion.div>

        {/* File Types Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="bg-white p-6 rounded-lg shadow-sm border border-gray-200"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">File Types</h3>
            <DocumentTextIcon className="h-5 w-5 text-gray-400" />
          </div>
          
          {fileTypeData.length > 0 ? (
            <div className="space-y-3">
              {fileTypeData.map(([type, count], index) => {
                const colors = ['bg-blue-500', 'bg-green-500', 'bg-purple-500']
                const color = colors[index % colors.length]
                return (
                  <div key={type} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${color}`} />
                      <span className="text-sm text-gray-700 uppercase">{type}</span>
                    </div>
                    <span className="text-sm font-medium text-gray-900">{count}</span>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              <p className="text-sm">No file type data</p>
            </div>
          )}
        </motion.div>
      </div>

      {/* Expiring Contracts Timeline */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="bg-white p-6 rounded-lg shadow-sm border border-gray-200"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold text-gray-900">Contract Expiration Timeline</h3>
            <CalendarIcon className="h-5 w-5 text-gray-400" />
          </div>
          
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-600">Show contracts expiring in:</label>
            <select
              value={selectedDays}
              onChange={(e) => setSelectedDays(Number(e.target.value))}
              className="text-sm border border-gray-300 rounded px-3 py-1 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value={30}>30 days</option>
              <option value={60}>60 days</option>
              <option value={90}>90 days</option>
              <option value={180}>6 months</option>
            </select>
          </div>
        </div>
        
        {!expiringLoading && dashboardData?.timeline ? (
          dashboardData.timeline.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {dashboardData.timeline.slice(0, 6).map((contract) => {
                const daysLeft = contract.days_until_expiry
                const urgencyColor = daysLeft <= 7 ? 'text-red-600' : daysLeft <= 30 ? 'text-orange-600' : 'text-green-600'
                const urgencyBg = daysLeft <= 7 ? 'bg-red-50 border-red-200' : daysLeft <= 30 ? 'bg-orange-50 border-orange-200' : 'bg-green-50 border-green-200'
                
                return (
                  <div key={contract.contract_id} className={`p-4 rounded-lg border ${urgencyBg}`}>
                    <div className="space-y-2">
                      <div className="flex items-start justify-between">
                        <div className="min-w-0 flex-1">
                          <p className="text-sm font-medium text-gray-900 truncate">
                            Contract #{contract.contract_id.slice(0, 8)}
                          </p>
                          <p className="text-xs text-gray-500">{contract.department}</p>
                        </div>
                        <div className={`text-xs font-medium ${urgencyColor} text-right`}>
                          {daysLeft > 0 ? (
                            <span>{daysLeft} days</span>
                          ) : daysLeft === 0 ? (
                            <span>Today!</span>
                          ) : (
                            <span>Expired</span>
                          )}
                        </div>
                      </div>
                      <div className="text-xs text-gray-500">
                        Expires: {new Date(contract.end_date).toLocaleDateString()}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="text-center text-gray-500 py-12">
              <CheckCircleIcon className="h-12 w-12 mx-auto mb-3 text-green-400" />
              <p className="font-medium">No contracts expiring soon</p>
              <p className="text-sm">All contracts are up to date within the selected timeframe</p>
            </div>
          )
        ) : expiringLoading ? (
          <div className="flex justify-center py-12">
            <LoadingSpinner />
          </div>
        ) : (
          <div className="text-center text-gray-500 py-12">
            <XCircleIcon className="h-12 w-12 mx-auto mb-3 text-gray-300" />
            <p className="font-medium">Failed to load timeline data</p>
            <button 
              onClick={handleRefresh}
              className="text-sm text-primary-600 hover:text-primary-700 mt-2"
            >
              Try again
            </button>
          </div>
        )}
      </motion.div>

      {/* System Health and Conflicts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Health */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="bg-white p-6 rounded-lg shadow-sm border border-gray-200"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">System Health</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${
                  stats.system_status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                }`} />
                <span className="text-sm text-gray-700 font-medium">
                  System Status: <span className="capitalize">{stats.system_status}</span>
                </span>
              </div>
              <CheckCircleIcon className="h-5 w-5 text-green-600" />
            </div>
            
            <div className="grid grid-cols-2 gap-4 text-center">
              <div className="p-3 bg-blue-50 rounded-lg">
                <p className="text-lg font-semibold text-blue-600">{stats.total_chunks}</p>
                <p className="text-xs text-blue-600">Text Chunks</p>
              </div>
              <div className="p-3 bg-purple-50 rounded-lg">
                <p className="text-lg font-semibold text-purple-600">GPT-4</p>
                <p className="text-xs text-purple-600">AI Model</p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Conflicts Overview */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="bg-white p-6 rounded-lg shadow-sm border border-gray-200"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Contract Conflicts</h3>
            <ExclamationTriangleIcon className="h-5 w-5 text-gray-400" />
          </div>
          
          {!conflictsLoading ? (
            <div className="space-y-4">
              {conflictsData && conflictsData.count > 0 ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-red-50 rounded-lg border border-red-200">
                    <span className="text-sm text-red-700 font-medium">
                      {conflictsData.count} conflicts detected
                    </span>
                    <ExclamationTriangleIcon className="h-5 w-5 text-red-600" />
                  </div>
                  <p className="text-xs text-gray-500">
                    Review conflicts in contract data that may need attention
                  </p>
                </div>
              ) : (
                <div className="text-center py-6">
                  <CheckCircleIcon className="h-8 w-8 mx-auto mb-2 text-green-400" />
                  <p className="text-sm font-medium text-gray-900">No conflicts detected</p>
                  <p className="text-xs text-gray-500">All contract data appears consistent</p>
                </div>
              )}
            </div>
          ) : (
            <div className="flex justify-center py-6">
              <LoadingSpinner />
            </div>
          )}
        </motion.div>
      </div>
    </div>
  )
}