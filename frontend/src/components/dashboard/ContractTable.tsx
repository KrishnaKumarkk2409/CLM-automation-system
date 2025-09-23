'use client'

import { useDocumentsList } from '@/hooks/use-api'

export default function ContractTable() {
  const { data } = useDocumentsList({ limit: 20, offset: 0 })
  const rows = data?.documents || []

  const badge = (status: string) => {
    const map: Record<string, string> = {
      active: 'bg-green-100 text-green-700',
      expiring: 'bg-orange-100 text-orange-700',
      expired: 'bg-red-100 text-red-700',
    }
    return map[status?.toLowerCase()] || 'bg-gray-100 text-gray-700'
  }

  return (
    <div className="card overflow-hidden">
      <div className="px-4 py-3 border-b border-secondary-200">
        <h3 className="text-sm font-semibold text-secondary-900">Recent Contracts</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="text-left text-secondary-500 border-b border-secondary-200">
              <th className="px-4 py-3">Contract</th>
              <th className="px-4 py-3">Company</th>
              <th className="px-4 py-3">Type</hth>
              <th className="px-4 py-3">Status</th>
              <th className="px-4 py-3">Expiration</th>
              <th className="px-4 py-3">Actions</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((d: any, i: number) => (
              <tr key={i} className="border-b border-secondary-100">
                <td className="px-4 py-3 text-secondary-900 truncate">{d.filename}</td>
                <td className="px-4 py-3 text-secondary-700">{d.metadata?.company || '—'}</td>
                <td className="px-4 py-3 uppercase text-secondary-600">{d.file_type}</td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-1 rounded text-xs ${badge(d.metadata?.status || 'active')}`}>
                    {(d.metadata?.status || 'Active').toString()}
                  </span>
                </td>
                <td className="px-4 py-3 text-secondary-600">{d.metadata?.end_date ? new Date(d.metadata.end_date).toLocaleDateString() : '—'}</td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2 text-secondary-600">
                    <button className="hover:text-secondary-900">View</button>
                    <span>•</span>
                    <button className="hover:text-secondary-900">Edit</button>
                  </div>
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td className="px-4 py-8 text-center text-secondary-500" colSpan={6}>No contracts found</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}


