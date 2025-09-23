'use client'

import { motion } from 'framer-motion'
import { 
  DocumentCheckIcon,
  CalendarIcon,
  BuildingOfficeIcon,
  ExclamationTriangleIcon,
  EyeIcon
} from '@heroicons/react/24/outline'
import { useContractsList } from '@/hooks/use-api'

export default function ContractsList() {
  const { data, isLoading, error } = useContractsList()
  const contracts = data?.contracts || []

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Active':
        return 'text-green-600 bg-green-50 border-green-200'
      case 'Expiring Soon':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'Expired':
        return 'text-red-600 bg-red-50 border-red-200'
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const getDepartmentColor = (department: string) => {
    switch (department) {
      case 'IT':
        return 'text-blue-600 bg-blue-50 border-blue-200'
      case 'Legal':
        return 'text-purple-600 bg-purple-50 border-purple-200'
      case 'Finance':
        return 'text-green-600 bg-green-50 border-green-200'
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  return (
    <div className="h-full overflow-auto bg-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-black mb-2">Contracts</h1>
          <p className="text-gray-600">View and manage your active contracts</p>
        </motion.div>

        {/* Contracts Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card"
        >
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Contract
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Parties
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    End Date
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Department
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Value
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {contracts.map((contract: any, index: number) => (
                  <motion.tr
                    key={contract.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 + index * 0.05 }}
                    className="hover:bg-gray-50"
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <DocumentCheckIcon className="h-5 w-5 text-gray-400 mr-3" />
                        <div>
                          <div className="text-sm font-medium text-black">{contract.name}</div>
                          <div className="text-xs text-gray-500">ID: {contract.id}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="text-sm text-gray-900">
                        {contract.parties.map((party: string, idx: number) => (
                          <div key={idx} className="flex items-center mb-1">
                            <BuildingOfficeIcon className="h-3 w-3 text-gray-400 mr-1" />
                            <span className="text-xs">{party}</span>
                          </div>
                        ))}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center text-sm text-gray-600">
                        <CalendarIcon className="h-4 w-4 mr-1" />
                        <div>
                          <div>{contract.endDate}</div>
                          <div className="text-xs text-gray-400">
                            {contract.daysToExpiry < 30 ? (
                              <span className="text-red-500 flex items-center">
                                <ExclamationTriangleIcon className="h-3 w-3 mr-1" />
                                {contract.daysToExpiry} days left
                              </span>
                            ) : (
                              <span>{contract.daysToExpiry} days left</span>
                            )}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 text-xs font-medium border ${getDepartmentColor(contract.department)}`}>
                        {contract.department}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 font-medium">
                      {contract.value}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 text-xs font-medium border ${getStatusColor(contract.status)}`}>
                        {contract.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <button
                        className="p-1 text-gray-400 hover:text-black border border-gray-300 hover:border-gray-400 bg-white"
                        title="View Details"
                      >
                        <EyeIcon className="h-4 w-4" />
                      </button>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>

          {isLoading && (
            <div className="text-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-black mx-auto mb-4"></div>
              <p className="text-gray-500">Loading contracts...</p>
            </div>
          )}
          
          {error && (
            <div className="text-center py-12">
              <DocumentCheckIcon className="h-12 w-12 text-red-300 mx-auto mb-4" />
              <p className="text-red-500">Failed to load contracts</p>
            </div>
          )}
          
          {!isLoading && !error && contracts.length === 0 && (
            <div className="text-center py-12">
              <DocumentCheckIcon className="h-12 w-12 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500">No contracts found</p>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  )
}