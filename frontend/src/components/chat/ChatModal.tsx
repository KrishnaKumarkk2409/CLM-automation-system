'use client'

import ChatInterface from './chat-interface'

export default function ChatModal({ open, onClose }: { open: boolean, onClose: () => void }) {
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="relative bg-white w-full max-w-md h-[600px] rounded-xl shadow-xl border border-secondary-200 overflow-hidden">
        <div className="absolute top-2 right-2">
          <button onClick={onClose} className="text-secondary-600 hover:text-secondary-900">✕</button>
        </div>
        <ChatInterface />
      </div>
    </div>
  )
}


