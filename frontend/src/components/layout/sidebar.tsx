'use client'

import { motion } from 'framer-motion'
import { XMarkIcon } from '@heroicons/react/24/outline'

interface SidebarProps {
  currentView: string
  onViewChange: (view: string) => void
  onClose: () => void
}

export default function Sidebar({ currentView, onViewChange, onClose }: SidebarProps) {
  return (
    <div className="h-full bg-white/90 backdrop-blur-sm border-r border-secondary-200/50 p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-secondary-900">Tools & Actions</h2>
        <button onClick={onClose} className="p-1 hover:bg-secondary-100 rounded">
          <XMarkIcon className="h-5 w-5" />
        </button>
      </div>
      
      <div className="space-y-4">
        <div className="text-sm text-secondary-600">
          Advanced features coming soon...
        </div>
      </div>
    </div>
  )
}