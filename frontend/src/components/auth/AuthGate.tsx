'use client'

import { useEffect, useState } from 'react'
import { supabase } from '@/lib/supabase'

export default function AuthGate({ children }: { children: React.ReactNode }) {
  const [isLoading, setIsLoading] = useState(true)
  const [isAuthed, setIsAuthed] = useState(false)

  useEffect(() => {
    let mounted = true
    const init = async () => {
      const { data } = await supabase.auth.getSession()
      if (!mounted) return
      setIsAuthed(!!data.session)
      setIsLoading(false)
    }
    init()
    const { data: sub } = supabase.auth.onAuthStateChange((_event, session) => {
      setIsAuthed(!!session)
    })
    return () => {
      mounted = false
      sub.subscription.unsubscribe()
    }
  }, [])

  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="text-gray-600">Loading...</div>
      </div>
    )
  }

  if (!isAuthed) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 p-6">
        <div className="bg-white border border-gray-200 p-8 max-w-sm w-full text-center">
          <h1 className="text-xl font-semibold text-black mb-2">Sign in to continue</h1>
          <p className="text-sm text-gray-600 mb-6">Use your Google account</p>
          <button
            onClick={async () => {
              await supabase.auth.signInWithOAuth({ provider: 'google', options: { redirectTo: window.location.origin } })
            }}
            className="w-full py-2 bg-black text-white hover:opacity-90"
          >
            Continue with Google
          </button>
        </div>
      </div>
    )
  }

  return <>{children}</>
}


