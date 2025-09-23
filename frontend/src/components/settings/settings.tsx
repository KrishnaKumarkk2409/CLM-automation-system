'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  CogIcon,
  ServerIcon,
  ChartBarIcon,
  BellIcon,
  UserIcon,
  ShieldCheckIcon,
  KeyIcon,
  PlusIcon,
  TrashIcon,
  ClipboardDocumentIcon,
  EyeIcon,
  EyeSlashIcon
} from '@heroicons/react/24/outline'
import { SystemStats, useAPIKeys, useCreateAPIKey, useDeleteAPIKey } from '@/hooks/use-api'
import StatsCards from '../dashboard/stats-cards'
import toast from 'react-hot-toast'

interface SettingsProps {
  stats?: SystemStats
}

export default function Settings({ stats }: SettingsProps) {
  const [showCreateKey, setShowCreateKey] = useState(false)
  const [newKeyName, setNewKeyName] = useState('')
  const [newKeyDescription, setNewKeyDescription] = useState('')
  const [showKeyValue, setShowKeyValue] = useState<string | null>(null)
  const [createdKey, setCreatedKey] = useState<string | null>(null)

  const { data: apiKeysData, refetch: refetchKeys } = useAPIKeys()
  const createKeyMutation = useCreateAPIKey()
  const deleteKeyMutation = useDeleteAPIKey()

  const handleCreateKey = async () => {
    if (!newKeyName.trim()) {
      toast.error('Please enter a key name')
      return
    }

    try {
      const result = await createKeyMutation.mutateAsync({
        name: newKeyName.trim(),
        description: newKeyDescription.trim()
      })
      
      setCreatedKey(result.key)
      setNewKeyName('')
      setNewKeyDescription('')
      setShowCreateKey(false)
      toast.success('API key created successfully!')
    } catch (error) {
      toast.error('Failed to create API key')
    }
  }

  const handleDeleteKey = async (keyId: string, keyName: string) => {
    if (!confirm(`Are you sure you want to delete the API key "${keyName}"? This action cannot be undone.`)) {
      return
    }

    try {
      await deleteKeyMutation.mutateAsync(keyId)
      toast.success('API key deleted successfully')
    } catch (error) {
      toast.error('Failed to delete API key')
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    toast.success('Copied to clipboard!')
  }

  const settingSections = [
    {
      title: 'Account Settings',
      icon: UserIcon,
      items: [
        { label: 'Profile Information', description: 'Update your personal information' },
        { label: 'Change Password', description: 'Update your account password' },
        { label: 'Email Preferences', description: 'Manage email notifications' }
      ]
    },
    {
      title: 'System Configuration',
      icon: ServerIcon,
      items: [
        { label: 'AI Model Settings', description: 'Configure AI analysis parameters' },
        { label: 'Document Processing', description: 'Set chunk size and processing options' },
        { label: 'Database Connection', description: 'Manage database settings' }
      ]
    },
    {
      title: 'Notifications',
      icon: BellIcon,
      items: [
        { label: 'Contract Expiry Alerts', description: 'Set up contract expiration notifications' },
        { label: 'Report Schedule', description: 'Configure automated report generation' },
        { label: 'System Alerts', description: 'Manage system status notifications' }
      ]
    },
    {
      title: 'Security',
      icon: ShieldCheckIcon,
      items: [
        { label: 'Access Control', description: 'Manage user permissions and roles' },
        { label: 'Audit Logs', description: 'View system access and activity logs' },
        { label: 'Data Backup', description: 'Configure data backup and recovery' }
      ]
    },
    {
      title: 'API Integration',
      icon: KeyIcon,
      items: [
        { label: 'API Keys', description: 'Manage API keys for third-party integrations' },
        { label: 'Webhooks', description: 'Configure webhook endpoints and events' },
        { label: 'Rate Limiting', description: 'Configure API rate limits and quotas' }
      ]
    }
  ]

  return (
    <div className="h-full overflow-auto bg-white p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-black mb-2">Settings</h1>
          <p className="text-gray-600">Configure your CLM automation system</p>
        </motion.div>

        {/* System Metrics */}
        {stats && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="mb-8"
          >
            <h2 className="text-xl font-semibold text-black mb-4 flex items-center">
              <ChartBarIcon className="h-5 w-5 mr-2" />
              System Metrics
            </h2>
            <div className="card p-6">
              <StatsCards stats={stats} />
            </div>
          </motion.div>
        )}

        {/* Settings Sections */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {settingSections.map((section, index) => {
            const IconComponent = section.icon
            return (
              <motion.div
                key={section.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 + index * 0.05 }}
                className="card p-6"
              >
                <div className="flex items-center mb-4">
                  <div className="p-2 bg-gray-100 border border-gray-200 mr-3">
                    <IconComponent className="h-5 w-5 text-black" />
                  </div>
                  <h3 className="text-lg font-semibold text-black">{section.title}</h3>
                </div>
                <div className="space-y-3">
                  {section.items.map((item, itemIndex) => (
                    <motion.button
                      key={itemIndex}
                      whileHover={{ x: 4 }}
                      className="w-full text-left p-3 border border-gray-200 hover:border-gray-300 hover:bg-gray-50 transition-all duration-200"
                    >
                      <div className="text-sm font-medium text-black">{item.label}</div>
                      <div className="text-xs text-gray-600 mt-1">{item.description}</div>
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            )
          })}
        </div>

        {/* System Information */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mt-8 card p-6"
        >
          <h3 className="text-lg font-semibold text-black mb-4 flex items-center">
            <CogIcon className="h-5 w-5 mr-2" />
            System Information
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <div className="font-medium text-black">Version</div>
              <div className="text-gray-600">CLM v1.0.0</div>
            </div>
            <div>
              <div className="font-medium text-black">Last Updated</div>
              <div className="text-gray-600">{new Date().toLocaleDateString()}</div>
            </div>
            <div>
              <div className="font-medium text-black">Environment</div>
              <div className="text-gray-600">Development</div>
            </div>
          </div>
        </motion.div>
        
        {/* API Key Management Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="mt-8 card p-6"
        >
          <div className="flex justify-between items-center mb-6">
            <div>
              <h3 className="text-lg font-semibold text-black flex items-center">
                <KeyIcon className="h-5 w-5 mr-2" />
                API Key Management
              </h3>
              <p className="text-sm text-gray-600 mt-1">
                Generate and manage API keys for third-party integrations
              </p>
            </div>
            <button
              onClick={() => setShowCreateKey(true)}
              className="btn-primary flex items-center space-x-2"
            >
              <PlusIcon className="h-4 w-4" />
              <span>Create API Key</span>
            </button>
          </div>
          
          {/* Created Key Display */}
          {createdKey && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg"
            >
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <h4 className="text-sm font-medium text-green-800 mb-2">
                    🎉 API Key Created Successfully!
                  </h4>
                  <p className="text-xs text-green-700 mb-3">
                    Make sure to copy your API key now. You won't be able to see it again!
                  </p>
                  <div className="flex items-center space-x-2">
                    <code className="text-xs bg-white px-3 py-2 border border-green-300 rounded font-mono flex-1">
                      {createdKey}
                    </code>
                    <button
                      onClick={() => copyToClipboard(createdKey)}
                      className="p-2 text-green-700 hover:text-green-800 hover:bg-green-100 rounded transition-colors"
                      title="Copy to clipboard"
                    >
                      <ClipboardDocumentIcon className="h-4 w-4" />
                    </button>
                  </div>
                </div>
                <button
                  onClick={() => setCreatedKey(null)}
                  className="text-green-600 hover:text-green-800 text-sm"
                >
                  ✕
                </button>
              </div>
            </motion.div>
          )}
          
          {/* API Usage Information */}
          <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h4 className="text-sm font-medium text-blue-800 mb-2">
              📚 API Usage Instructions
            </h4>
            <div className="text-xs text-blue-700 space-y-2">
              <p><strong>Upload Endpoint:</strong> <code>POST /api/upload</code></p>
              <p><strong>Authentication:</strong> Include your API key in the request</p>
              <div className="mt-2 bg-white p-2 rounded border">
                <pre className="text-xs font-mono text-gray-800">{`curl -X POST http://localhost:8000/api/upload \
  -H "Content-Type: multipart/form-data" \
  -F "api_key=your_api_key_here" \
  -F "files=@document.pdf"`}</pre>
              </div>
            </div>
          </div>
          
          {/* API Keys List */}
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-gray-900">Active API Keys</h4>
            {apiKeysData?.api_keys?.length > 0 ? (
              apiKeysData.api_keys.map((key: any) => (
                <motion.div
                  key={key.id}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:border-gray-300 transition-colors"
                >
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <h5 className="text-sm font-medium text-black">{key.name}</h5>
                      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                        key.active ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                      }`}>
                        {key.active ? 'Active' : 'Inactive'}
                      </span>
                    </div>
                    <p className="text-xs text-gray-600 mt-1">{key.description}</p>
                    <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                      <span>Created: {new Date(key.created_at).toLocaleDateString()}</span>
                      {key.last_used && (
                        <span>Last used: {new Date(key.last_used).toLocaleDateString()}</span>
                      )}
                      {key.expires_at && (
                        <span>Expires: {new Date(key.expires_at).toLocaleDateString()}</span>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => handleDeleteKey(key.id, key.name)}
                    className="p-2 text-red-600 hover:text-red-800 hover:bg-red-50 rounded transition-colors"
                    title="Delete API key"
                  >
                    <TrashIcon className="h-4 w-4" />
                  </button>
                </motion.div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                <KeyIcon className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                <p className="text-sm font-medium mb-2">No API keys created yet</p>
                <p className="text-xs">Create your first API key to enable third-party integrations</p>
              </div>
            )}
          </div>
        </motion.div>
        
        {/* Create API Key Modal */}
        <AnimatePresence>
          {showCreateKey && (
            <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="bg-white rounded-lg max-w-md w-full p-6"
              >
                <h3 className="text-lg font-semibold text-black mb-4">
                  Create New API Key
                </h3>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Name *
                    </label>
                    <input
                      type="text"
                      value={newKeyName}
                      onChange={(e) => setNewKeyName(e.target.value)}
                      className="input-field w-full"
                      placeholder="e.g., Mobile App, Third Party Integration"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Description
                    </label>
                    <textarea
                      value={newKeyDescription}
                      onChange={(e) => setNewKeyDescription(e.target.value)}
                      className="input-field w-full h-20 resize-none"
                      placeholder="Optional description for this API key"
                    />
                  </div>
                  
                  <div className="text-xs text-gray-600">
                    <p>⚠️ API keys provide full access to your system. Keep them secure!</p>
                  </div>
                </div>
                
                <div className="flex justify-end space-x-3 mt-6">
                  <button
                    onClick={() => {
                      setShowCreateKey(false)
                      setNewKeyName('')
                      setNewKeyDescription('')
                    }}
                    className="btn-ghost"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleCreateKey}
                    disabled={!newKeyName.trim() || createKeyMutation.isPending}
                    className="btn-primary disabled:opacity-50"
                  >
                    {createKeyMutation.isPending ? 'Creating...' : 'Create API Key'}
                  </button>
                </div>
              </motion.div>
            </div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}