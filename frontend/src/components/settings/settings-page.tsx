'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  CogIcon,
  EnvelopeIcon,
  ServerIcon,
  KeyIcon,
  DocumentTextIcon,
  CheckCircleIcon,
  XCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline'
import { 
  useSystemConfig, 
  useUpdateEmailConfig,
  useProcessFolderDocuments 
} from '@/hooks/use-api'
import toast from 'react-hot-toast'
import LoadingSpinner from '@/components/ui/loading-spinner'

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState('email')
  const [emailForm, setEmailForm] = useState({
    smtp_server: '',
    smtp_port: 587,
    email_username: '',
    email_password: ''
  })
  
  const { data: config, isLoading: configLoading, refetch: refetchConfig } = useSystemConfig()
  const updateEmail = useUpdateEmailConfig()
  const processFolder = useProcessFolderDocuments()

  const handleSaveEmailConfig = async () => {
    try {
      await updateEmail.mutateAsync(emailForm)
      toast.success('Email configuration updated successfully!')
      refetchConfig()
    } catch (error) {
      toast.error('Failed to update email configuration')
    }
  }

  const handleProcessFolder = async () => {
    try {
      const result = await processFolder.mutateAsync()
      toast.success(`Processed ${result.total_processed} documents successfully!`)
      if (result.total_failed > 0) {
        toast.error(`Failed to process ${result.total_failed} documents`)
      }
    } catch (error) {
      toast.error('Failed to process folder documents')
    }
  }

  const tabs = [
    { id: 'email', label: 'Email & SMTP', icon: EnvelopeIcon },
    { id: 'system', label: 'System Status', icon: ServerIcon },
    { id: 'processing', label: 'Document Processing', icon: DocumentTextIcon },
    { id: 'security', label: 'API & Security', icon: KeyIcon }
  ]

  if (configLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  const renderEmailSettings = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">SMTP Configuration</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              SMTP Server
            </label>
            <input
              type="text"
              value={emailForm.smtp_server}
              onChange={(e) => setEmailForm(prev => ({ ...prev, smtp_server: e.target.value }))}
              placeholder="smtp.gmail.com"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              SMTP Port
            </label>
            <input
              type="number"
              value={emailForm.smtp_port}
              onChange={(e) => setEmailForm(prev => ({ ...prev, smtp_port: Number(e.target.value) }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Email Username
            </label>
            <input
              type="email"
              value={emailForm.email_username}
              onChange={(e) => setEmailForm(prev => ({ ...prev, email_username: e.target.value }))}
              placeholder="your-email@gmail.com"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Email Password / App Password
            </label>
            <input
              type="password"
              value={emailForm.email_password}
              onChange={(e) => setEmailForm(prev => ({ ...prev, email_password: e.target.value }))}
              placeholder="Enter app password"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
        </div>
        
        <div className="mt-4 p-4 bg-blue-50 rounded-lg">
          <div className="flex items-start gap-2">
            <InformationCircleIcon className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-blue-800">
              <p className="font-medium">Gmail Setup Instructions:</p>
              <ul className="mt-1 list-disc list-inside space-y-1">
                <li>Use your Gmail address as username</li>
                <li>Generate an App Password in your Google Account settings</li>
                <li>Use the App Password instead of your regular password</li>
                <li>Ensure 2-factor authentication is enabled</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div className="mt-6">
          <button
            onClick={handleSaveEmailConfig}
            disabled={updateEmail.isPending}
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {updateEmail.isPending ? 'Saving...' : 'Save Email Configuration'}
          </button>
        </div>
      </div>
    </div>
  )

  const renderSystemStatus = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">System Configuration Status</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            {
              label: 'SMTP Configuration',
              status: config?.smtp_configured,
              description: 'Email sending for reports'
            },
            {
              label: 'OpenAI Integration',
              status: config?.openai_configured,
              description: 'AI processing and chat functionality'
            },
            {
              label: 'Supabase Database',
              status: config?.supabase_configured,
              description: 'Document storage and search'
            }
          ].map((item, index) => (
            <div key={index} className="p-4 border border-gray-200 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-medium text-gray-900">{item.label}</h4>
                  <p className="text-sm text-gray-600">{item.description}</p>
                </div>
                <div className="flex-shrink-0">
                  {item.status ? (
                    <CheckCircleIcon className="h-6 w-6 text-green-600" />
                  ) : (
                    <XCircleIcon className="h-6 w-6 text-red-600" />
                  )}
                </div>
              </div>
              <div className="mt-2">
                <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                  item.status 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-red-100 text-red-800'
                }`}>
                  {item.status ? 'Configured' : 'Not Configured'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">System Information</h3>
        
        <div className="bg-gray-50 p-4 rounded-lg space-y-2">
          <div className="flex justify-between">
            <span className="text-sm text-gray-600">System Version:</span>
            <span className="text-sm font-medium text-gray-900">{config?.system_version}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-sm text-gray-600">Documents Folder:</span>
            <span className="text-sm font-medium text-gray-900">{config?.documents_folder}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-sm text-gray-600">Chunk Size:</span>
            <span className="text-sm font-medium text-gray-900">{config?.chunk_size} characters</span>
          </div>
          <div className="flex justify-between">
            <span className="text-sm text-gray-600">Similarity Threshold:</span>
            <span className="text-sm font-medium text-gray-900">{config?.similarity_threshold}</span>
          </div>
        </div>
      </div>
    </div>
  )

  const renderProcessingSettings = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Document Processing</h3>
        
        <div className="space-y-4">
          <div className="p-4 border border-gray-200 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-2">Process Folder Documents</h4>
            <p className="text-sm text-gray-600 mb-4">
              Process all documents in the configured documents folder. This will analyze new documents and update the system.
            </p>
            <button
              onClick={handleProcessFolder}
              disabled={processFolder.isPending}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {processFolder.isPending ? (
                <>
                  <LoadingSpinner size="sm" />
                  Processing...
                </>
              ) : (
                <>
                  <DocumentTextIcon className="h-4 w-4" />
                  Process Folder
                </>
              )}
            </button>
          </div>
          
          <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-start gap-2">
              <ExclamationTriangleIcon className="h-5 w-5 text-yellow-600 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-yellow-800">
                <p className="font-medium">Processing Guidelines:</p>
                <ul className="mt-1 list-disc list-inside space-y-1">
                  <li>Ensure documents are in supported formats (PDF, DOCX, TXT)</li>
                  <li>Large documents may take longer to process</li>
                  <li>OCR processing is automatic for scanned documents</li>
                  <li>Processed documents will be available for search and chat</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )

  const renderSecuritySettings = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">API & Security Configuration</h3>
        
        <div className="space-y-4">
          <div className="p-4 border border-gray-200 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <KeyIcon className="h-5 w-5 text-gray-600" />
              <h4 className="font-medium text-gray-900">API Keys Status</h4>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">OpenAI API Key:</span>
                <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                  config?.openai_configured 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-red-100 text-red-800'
                }`}>
                  {config?.openai_configured ? 'Configured' : 'Missing'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Supabase Keys:</span>
                <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                  config?.supabase_configured 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-red-100 text-red-800'
                }`}>
                  {config?.supabase_configured ? 'Configured' : 'Missing'}
                </span>
              </div>
            </div>
          </div>
          
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-start gap-2">
              <ExclamationTriangleIcon className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-red-800">
                <p className="font-medium">Security Notice:</p>
                <ul className="mt-1 list-disc list-inside space-y-1">
                  <li>API keys are stored securely and not displayed in the UI</li>
                  <li>Update API keys through environment variables or deployment configuration</li>
                  <li>Never share API keys or include them in code repositories</li>
                  <li>Regularly rotate API keys for enhanced security</li>
                </ul>
              </div>
            </div>
          </div>
          
          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-start gap-2">
              <InformationCircleIcon className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-blue-800">
                <p className="font-medium">Environment Variables:</p>
                <ul className="mt-1 list-disc list-inside space-y-1">
                  <li><code>OPENAI_API_KEY</code> - Required for AI functionality</li>
                  <li><code>SUPABASE_URL</code> - Database connection</li>
                  <li><code>SUPABASE_KEY</code> - Database authentication</li>
                  <li><code>EMAIL_USERNAME</code> - SMTP email address</li>
                  <li><code>EMAIL_PASSWORD</code> - SMTP app password</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )

  return (
    <div className="p-4 lg:p-8 max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl lg:text-3xl font-bold text-gray-900">System Settings</h1>
        <p className="text-gray-600 mt-1">Configure your CLM automation system</p>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Sidebar */}
        <div className="lg:w-64 flex-shrink-0">
          <nav className="space-y-1">
            {tabs.map((tab) => {
              const IconComponent = tab.icon
              return (
                <motion.button
                  key={tab.id}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
                    activeTab === tab.id
                      ? 'bg-primary-100 text-primary-700 border border-primary-200'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
                >
                  <IconComponent className="mr-3 h-5 w-5" />
                  {tab.label}
                </motion.button>
              )
            })}
          </nav>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3 }}
            className="bg-white rounded-lg border border-gray-200 p-6"
          >
            {activeTab === 'email' && renderEmailSettings()}
            {activeTab === 'system' && renderSystemStatus()}
            {activeTab === 'processing' && renderProcessingSettings()}
            {activeTab === 'security' && renderSecuritySettings()}
          </motion.div>
        </div>
      </div>
    </div>
  )
}