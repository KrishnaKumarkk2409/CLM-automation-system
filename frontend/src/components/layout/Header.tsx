'use client'

import { useState } from 'react'
import { BellIcon, Cog6ToothIcon, MagnifyingGlassIcon, UserCircleIcon } from '@heroicons/react/24/outline'
import { useAdvancedDocumentSearch } from '@/hooks/use-api'
import toast from 'react-hot-toast'

interface HeaderProps {
  onSearchResults?: (results: any[]) => void
}

export default function Header({ onSearchResults }: HeaderProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const advancedSearch = useAdvancedDocumentSearch()
  
  const runHeaderSearch = async () => {
    if (!searchTerm.trim()) return
    try {
      const data = await advancedSearch.mutateAsync({ query: searchTerm, limit: 12 })
      if (onSearchResults) {
        onSearchResults(data.documents || [])
      }
    } catch (e) {
      toast.error('Search failed')
    }
  }

  return (
    <header className="sticky top-0 z-10 bg-white/80 backdrop-blur border-b border-secondary-200/50 px-3 sm:px-4 lg:px-6 py-2 sm:py-3">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Left: Logo + Title */}
        <div className="flex items-center space-x-2 sm:space-x-3">
          <div className="h-7 w-7 sm:h-8 sm:w-8 rounded bg-primary flex items-center justify-center">
            <span className="text-white font-semibold text-xs sm:text-sm">CLM</span>
          </div>
          <span className="font-medium text-foreground text-sm sm:text-base">Contract Manager</span>
        </div>

        {/* Right: Search + Icons */}
        <div className="flex items-center space-x-2 sm:space-x-4">
          {/* Search Input */}
          <div className="relative w-[180px] xs:w-[240px] sm:w-[300px] md:w-[380px] hidden xs:block">
            <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
              <MagnifyingGlassIcon className="h-3.5 w-3.5 sm:h-4 sm:w-4 text-secondary-400" />
            </div>
            <input
              type="text"
              className="input-field text-xs sm:text-sm py-1.5 sm:py-2 pl-8 sm:pl-10"
              placeholder="Search contracts..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && runHeaderSearch()}
            />
          </div>

          {/* Mobile Search Button */}
          <button className="xs:hidden p-1.5 sm:p-2 text-secondary-600 hover:text-secondary-900 rounded-full hover:bg-secondary-100">
            <MagnifyingGlassIcon className="h-5 w-5" />
          </button>

          {/* Notification Bell */}
          <button className="relative p-1.5 sm:p-2 text-secondary-600 hover:text-secondary-900 rounded-full hover:bg-secondary-100">
            <BellIcon className="h-4.5 w-4.5 sm:h-5 sm:w-5" />
            <span className="absolute top-0.5 sm:top-1 right-0.5 sm:right-1 h-3.5 w-3.5 sm:h-4 sm:w-4 bg-destructive text-white text-[10px] sm:text-xs flex items-center justify-center rounded-full">3</span>
          </button>

          {/* Settings */}
          <button className="p-1.5 sm:p-2 text-secondary-600 hover:text-secondary-900 rounded-full hover:bg-secondary-100">
            <Cog6ToothIcon className="h-4.5 w-4.5 sm:h-5 sm:w-5" />
          </button>

          {/* User */}
          <button className="p-1.5 sm:p-2 text-secondary-600 hover:text-secondary-900 rounded-full hover:bg-secondary-100">
            <UserCircleIcon className="h-4.5 w-4.5 sm:h-5 sm:w-5" />
          </button>
        </div>
      </div>
    </header>
  )
}