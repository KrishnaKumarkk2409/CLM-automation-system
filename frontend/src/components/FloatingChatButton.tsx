'use client'

import { useState } from 'react'
import { ChatBubbleLeftRightIcon } from '@heroicons/react/24/solid'

export default function FloatingChatButton({ onClick }: { onClick: () => void }) {
  const [unread, setUnread] = useState(true) // Set to true to show the unread indicator
  return (
    <button
      onClick={() => { setUnread(false); onClick() }}
      className="fixed bottom-6 right-6 z-50 h-14 w-14 rounded-full bg-primary hover:bg-primary-hover text-white shadow-lg flex items-center justify-center"
      aria-label="Open chat"
    >
      <ChatBubbleLeftRightIcon className="h-6 w-6" />
      {unread && (
        <span className="absolute -top-1 -right-1 h-5 w-5 rounded-full bg-destructive text-white text-xs flex items-center justify-center font-bold">!</span>
      )}
    </button>
  )
}


