'use client'

import { useContractConflicts, useExpiringContracts } from '@/hooks/use-api'

export default function AlertsPanel() {
  const { data: conflicts } = useContractConflicts()
  const { data: expiring } = useExpiringContracts(30)

  const items: Array<{ level: 'high'|'medium'|'low', title: string, subtitle: string, ts: string }> = []
  if (conflicts?.count) items.push({ level: 'high', title: 'Conflicts Detected', subtitle: `${conflicts.count} conflicts in contracts`, ts: 'Just now' })
  const expNum = expiring?.count || 0
  if (expNum) items.push({ level: 'medium', title: 'Contracts Expiring Soon', subtitle: `${expNum} contracts within 30 days`, ts: 'Today' })

  return (
    <div className="card">
      <div className="px-4 py-3 border-b border-secondary-200 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-secondary-900">Active Alerts</h3>
        {items.length > 0 && <span className="text-xs px-2 py-0.5 rounded-full bg-red-100 text-red-700">{items.length} Critical</span>}
      </div>
      <div className="p-4 space-y-3">
        {items.map((a, i) => (
          <div key={i} className={`p-3 rounded-lg border ${a.level==='high' ? 'bg-red-50 border-red-200' : a.level==='medium' ? 'bg-orange-50 border-orange-200' : 'bg-blue-50 border-blue-200'}`}>
            <p className={`text-sm font-medium ${a.level==='high' ? 'text-red-700' : a.level==='medium' ? 'text-orange-700' : 'text-blue-700'}`}>{a.title}</p>
            <p className={`text-xs ${a.level==='high' ? 'text-red-600' : a.level==='medium' ? 'text-orange-600' : 'text-blue-600'}`}>{a.subtitle}</p>
            <p className="text-xs text-secondary-400 mt-1">{a.ts}</p>
          </div>
        ))}
        {items.length === 0 && <p className="text-sm text-secondary-500">No active alerts</p>}
      </div>
    </div>
  )
}


