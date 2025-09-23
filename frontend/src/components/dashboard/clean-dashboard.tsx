'use client'

import { motion } from 'framer-motion'
import { 
  DocumentTextIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  ArrowUpOnSquareStackIcon,
  ChartBarIcon,
  EnvelopeIcon,
} from '@heroicons/react/24/outline'
import { 
  useSystemStats,
  useAnalyticsDashboard,
  useExpiringContracts,
  useContractConflicts,
  useDocumentsList,
  useGenerateAnalyticsReport,
} from '@/hooks/use-api'
import { useState } from 'react'
import toast from 'react-hot-toast'

interface CleanDashboardProps {
  onUpload?: () => void
  onAnalytics?: () => void
  onStartChat?: () => void
}

export default function CleanDashboard({ onUpload, onAnalytics, onStartChat }: CleanDashboardProps) {
  const { data: stats, isError: statsError } = useSystemStats({ retry: false, retryOnMount: false })
  const { data: analytics, isError: analyticsError } = useAnalyticsDashboard({ retry: false, retryOnMount: false })
  const { data: expiring, isError: expiringError } = useExpiringContracts(30, { retry: false, retryOnMount: false })
  const { data: conflicts, isError: conflictsError } = useContractConflicts({ retry: false, retryOnMount: false })
  const { data: docs, isError: docsError } = useDocumentsList({ limit: 5, offset: 0 }, { retry: false, retryOnMount: false })
  const generateReport = useGenerateAnalyticsReport()

  const [sending, setSending] = useState(false)

  const handleGenerateReport = async () => {
    try {
      setSending(true)
      await generateReport.mutateAsync("")
      toast.success('Report generated')
    } catch (e) {
      toast.error('Failed to generate report')
    } finally {
      setSending(false)
    }
  }

  const totalContracts = analytics?.overview?.active_contracts || 0
  const expiringCount = expiring?.count || (analytics?.timeline?.length || 0)
  const conflictsCount = conflicts?.count || 0
  const activeRenewals = Math.max(0, Math.round(totalContracts * 0.16))

  // Animation variants
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
    <div className="h-full overflow-auto bg-gradient-to-br from-gray-50 to-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-6 lg:py-8">
        {/* Hero */}
        <motion.div 
          className="mb-6 sm:mb-8 lg:mb-10"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary-600 to-accent-500 break-words">
            Contract Lifecycle Management
          </h1>
          <p className="text-gray-600 mt-2 text-base sm:text-lg max-w-full sm:max-w-3xl">
            Intelligently manage your contracts with automated insights, conflict detection, and AI-powered analysis
          </p>
        </motion.div>

        {/* Top Stats */}
        <motion.div 
          className="grid grid-cols-1 xs:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 lg:gap-5 mb-6 sm:mb-8 lg:mb-10"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {[
            { 
              label: 'Total Contracts', 
              value: totalContracts, 
              icon: DocumentTextIcon, 
              color: 'text-blue-600',
              bgColor: 'from-blue-50 to-blue-100',
              iconBg: 'bg-blue-100',
              borderColor: 'border-blue-200',
              pill: '+12%',
              pillColor: 'text-blue-700 bg-blue-100' 
            },
            { 
              label: 'Expiring Soon', 
              value: expiringCount, 
              icon: ClockIcon, 
              color: 'text-orange-600',
              bgColor: 'from-orange-50 to-orange-100',
              iconBg: 'bg-orange-100',
              borderColor: 'border-orange-200',
              sub: 'Next 30 days' 
            },
            { 
              label: 'Conflicts Detected', 
              value: conflictsCount, 
              icon: ExclamationTriangleIcon, 
              color: 'text-red-600',
              bgColor: 'from-red-50 to-red-100',
              iconBg: 'bg-red-100',
              borderColor: 'border-red-200',
              sub: 'Requires attention' 
            },
            { 
              label: 'Active Renewals', 
              value: activeRenewals, 
              icon: ChartBarIcon, 
              color: 'text-green-600',
              bgColor: 'from-green-50 to-green-100',
              iconBg: 'bg-green-100',
              borderColor: 'border-green-200',
              pill: '+5%',
              pillColor: 'text-green-700 bg-green-100' 
            },
          ].map((card, idx) => {
            const Icon = card.icon
            return (
              <motion.div
                key={card.label}
                variants={itemVariants}
                className={`bg-gradient-to-br ${card.bgColor} border ${card.borderColor} rounded-xl p-6 shadow-soft hover:shadow-soft-lg transition-all duration-300`}
                whileHover={{ y: -5, transition: { duration: 0.2 } }}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">{card.label}</p>
                    <div className="flex items-end gap-2 mt-2">
                      <p className={`text-3xl font-bold ${card.color}`}>{card.value}</p>
                      {card.pill && (
                        <span className={`text-xs px-2 py-1 rounded-full ${card.pillColor || 'text-gray-600 bg-gray-100'}`}>
                          {card.pill}
                        </span>
                      )}
                    </div>
                    {card.sub && <p className="text-xs text-gray-600 mt-1">{card.sub}</p>}
                  </div>
                  <div className={`p-3 rounded-full ${card.iconBg}`}>
                    <Icon className={`h-6 w-6 ${card.color}`} />
                  </div>
                </div>
              </motion.div>
            )
          })}
        </motion.div>

        {/* Quick Actions */}
        <motion.div 
          className="bg-white border border-gray-200 rounded-xl p-4 sm:p-6 shadow-soft mb-6 sm:mb-8 lg:mb-10"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.5 }}
        >
          <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-2 sm:mb-3">Quick Actions</h2>
          <p className="text-xs sm:text-sm text-gray-600 mb-4 sm:mb-5">Upload new contracts or manage existing ones</p>
          <div className="grid grid-cols-1 xs:grid-cols-2 md:flex md:flex-wrap gap-3 sm:gap-4">
            <button 
              onClick={onUpload} 
              className="inline-flex items-center justify-center gap-2 px-3 sm:px-5 py-2 sm:py-2.5 bg-gradient-to-r from-primary-600 to-primary-700 text-white text-sm sm:text-base rounded-lg shadow-sm hover:shadow-md transition-all duration-200"
            >
              <ArrowUpOnSquareStackIcon className="h-4 w-4 sm:h-5 sm:w-5" /> Upload Contract
            </button>
            <button 
              onClick={handleGenerateReport} 
              disabled={sending} 
              className="inline-flex items-center justify-center gap-2 px-3 sm:px-5 py-2 sm:py-2.5 bg-gradient-to-r from-gray-100 to-gray-200 text-gray-800 text-sm sm:text-base rounded-lg shadow-sm hover:shadow-md transition-all duration-200 disabled:opacity-50"
            >
              <EnvelopeIcon className="h-4 w-4 sm:h-5 sm:w-5" /> {sending ? 'Generating...' : 'Generate Report'}
            </button>
            <button 
              onClick={onAnalytics} 
              className="inline-flex items-center justify-center gap-2 px-3 sm:px-5 py-2 sm:py-2.5 bg-gradient-to-r from-gray-100 to-gray-200 text-gray-800 text-sm sm:text-base rounded-lg shadow-sm hover:shadow-md transition-all duration-200"
            >
              <ChartBarIcon className="h-4 w-4 sm:h-5 sm:w-5" /> View Analytics
            </button>
          </div>
        </motion.div>

        {/* Main Grid */}
        <motion.div 
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6, duration: 0.5 }}
        >
          {/* Recent Contracts */}
          <div className="md:col-span-2 bg-white border border-gray-200 rounded-xl shadow-soft overflow-hidden">
            <div className="p-3 sm:p-5 border-b border-gray-200 bg-gradient-to-r from-gray-50 to-white">
              <h3 className="text-lg sm:text-xl font-semibold text-gray-900">Recent Contracts</h3>
              <p className="text-xs sm:text-sm text-gray-600 mt-1">Overview of your contract portfolio with status indicators</p>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-500 border-b border-gray-200 bg-gray-50">
                    <th className="px-3 sm:px-5 py-2 sm:py-3 font-medium">Contract Name</th>
                    <th className="px-3 sm:px-5 py-2 sm:py-3 font-medium">Type</th>
                    <th className="hidden sm:table-cell px-3 sm:px-5 py-2 sm:py-3 font-medium">Department</th>
                    <th className="px-3 sm:px-5 py-2 sm:py-3 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {(docs?.documents || []).map((d: any, i: number) => (
                    <motion.tr 
                      key={d.id || i} 
                      className="border-b border-gray-100 hover:bg-gray-50 transition-colors"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.05 + 0.7 }}
                    >
                      <td className="px-3 sm:px-5 py-3 sm:py-4 text-gray-900 font-medium truncate max-w-[150px] sm:max-w-none">{d.filename}</td>
                      <td className="px-3 sm:px-5 py-3 sm:py-4 uppercase text-xs font-medium tracking-wider">
                        <span className="px-2 py-1 rounded-full bg-blue-50 text-blue-700">{d.file_type}</span>
                      </td>
                      <td className="hidden sm:table-cell px-3 sm:px-5 py-3 sm:py-4 text-gray-600">{d.metadata?.department || '—'}</td>
                      <td className="px-3 sm:px-5 py-3 sm:py-4">
                        <div className="flex items-center gap-2 sm:gap-3">
                          <button 
                            title="View" 
                            className="p-1 sm:p-1.5 rounded-full bg-gray-100 hover:bg-primary-50 text-gray-600 hover:text-primary-600 transition-colors"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3 sm:h-4 sm:w-4" viewBox="0 0 20 20" fill="currentColor">
                              <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
                              <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd" />
                            </svg>
                          </button>
                          <button 
                            title="Download" 
                            className="p-1 sm:p-1.5 rounded-full bg-gray-100 hover:bg-primary-50 text-gray-600 hover:text-primary-600 transition-colors"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3 sm:h-4 sm:w-4" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                            </svg>
                          </button>
                        </div>
                      </td>
                    </motion.tr>
                  ))}
                  {(!docs || docs.documents?.length === 0) && (
                    <tr>
                      <td className="px-3 sm:px-5 py-6 sm:py-8 text-center text-gray-500" colSpan={4}>
                        <div className="flex flex-col items-center">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 sm:h-10 sm:w-10 text-gray-300 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                          <p className="text-sm">No recent contracts</p>
                          <button 
                            onClick={onUpload}
                            className="mt-2 sm:mt-3 text-xs sm:text-sm text-primary-600 hover:text-primary-700 font-medium"
                          >
                            Upload your first contract
                          </button>
                        </div>
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Alerts */}
          <div className="bg-white border border-gray-200 rounded-xl shadow-soft overflow-hidden">
            <div className="p-3 sm:p-5 border-b border-gray-200 bg-gradient-to-r from-gray-50 to-white flex items-center justify-between">
              <h3 className="text-lg sm:text-xl font-semibold text-gray-900">Active Alerts</h3>
              {conflictsCount > 0 && (
                <span className="text-xs px-2 py-0.5 sm:px-2.5 sm:py-1 rounded-full bg-red-100 text-red-700 font-medium">{conflictsCount} Critical</span>
              )}
            </div>
            <div className="p-3 sm:p-5 space-y-3 sm:space-y-4">
              {conflictsCount > 0 && (
                <motion.div 
                  className="p-3 sm:p-4 rounded-xl border border-red-200 bg-gradient-to-r from-red-50 to-red-100"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.8 }}
                >
                  <div className="flex items-start">
                    <div className="p-1.5 sm:p-2 bg-red-200 rounded-full mr-2 sm:mr-3">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 sm:h-5 sm:w-5 text-red-600" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div>
                      <p className="text-xs sm:text-sm font-medium text-red-700">Conflicts Detected</p>
                      <p className="text-xs text-red-600 mt-0.5 sm:mt-1">{conflictsCount} conflicting items found in contracts</p>
                      <button className="mt-1.5 sm:mt-2 text-xs font-medium text-red-700 hover:text-red-800 inline-flex items-center">
                        Review Conflicts
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3 ml-1" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                        </svg>
                      </button>
                    </div>
                  </div>
                </motion.div>
              )}
              {(analytics?.timeline || []).slice(0, 3).map((t, idx) => (
                <motion.div 
                  key={idx} 
                  className="p-3 sm:p-4 rounded-xl border border-orange-200 bg-gradient-to-r from-orange-50 to-orange-100"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.8 + (idx * 0.1) }}
                >
                  <div className="flex items-start">
                    <div className="p-1.5 sm:p-2 bg-orange-200 rounded-full mr-2 sm:mr-3">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 sm:h-5 sm:w-5 text-orange-600" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div>
                      <p className="text-xs sm:text-sm font-medium text-orange-700">Contract Expiring Soon</p>
                      <p className="text-xs text-orange-600 mt-0.5 sm:mt-1">Contract {t.contract_id?.slice(0, 8)} expires {new Date(t.end_date).toLocaleDateString()}</p>
                      <button className="mt-1.5 sm:mt-2 text-xs font-medium text-orange-700 hover:text-orange-800 inline-flex items-center">
                        View Details
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3 ml-1" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                        </svg>
                      </button>
                    </div>
                  </div>
                </motion.div>
              ))}
              {conflictsCount === 0 && (analytics?.timeline?.length || 0) === 0 && (
                <div className="p-8 flex flex-col items-center justify-center text-center">
                  <div className="p-3 bg-green-100 rounded-full mb-3">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-green-600" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <p className="text-sm font-medium text-gray-700">All Clear</p>
                  <p className="text-xs text-gray-500 mt-1">No active alerts at this time</p>
                </div>
              )}
            </div>
          </div>
        </motion.div>

        {/* Assistant CTA */}
        <motion.div 
          className="mt-10 flex justify-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1, duration: 0.5 }}
        >
          <button 
            onClick={onStartChat} 
            className="group inline-flex items-center gap-3 px-6 py-3 bg-gradient-to-r from-gray-900 to-gray-800 text-white rounded-xl shadow-md hover:shadow-lg transition-all duration-200"
          >
            <span className="p-1.5 bg-white/10 rounded-full group-hover:bg-white/20 transition-colors">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z" />
                <path d="M15 7v2a4 4 0 01-4 4H9.828l-1.766 1.767c.28.149.599.233.938.233h2l3 3v-3h2a2 2 0 002-2V9a2 2 0 00-2-2h-1z" />
              </svg>
            </span>
            <span className="text-lg font-medium">Ask about contracts…</span>
            <span className="group-hover:translate-x-1 transition-transform">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
              </svg>
            </span>
          </button>
        </motion.div>
      </div>
    </div>
  )
}



