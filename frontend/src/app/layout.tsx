import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from '@/components/providers'
import dynamic from 'next/dynamic'
import { Toaster } from 'react-hot-toast'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'CLM Automation - Contract Lifecycle Management',
  description: 'AI-powered contract analysis and management system',
  keywords: 'contract management, AI, automation, legal tech, document analysis',
  authors: [{ name: 'CLM Team' }],
  metadataBase: new URL('http://localhost:3000'),
  openGraph: {
    title: 'CLM Automation System',
    description: 'AI-powered contract analysis and management',
    type: 'website',
    locale: 'en_US',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const AuthGate = dynamic(() => import('@/components/auth/AuthGate'), { ssr: false })
  return (
    <html lang="en" className="h-full">
      <head>
        <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
        <link rel="icon" type="image/png" href="/favicon.png" />
        <meta name="theme-color" content="#000000" />
      </head>
      <body className={`${inter.className} h-full antialiased`}>
        <Providers>
          <AuthGate>
            <div className="min-h-screen bg-white">
              {children}
            </div>
          </AuthGate>
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#fff',
                color: '#000',
                boxShadow: '0 2px 6px 0 rgba(0, 0, 0, 0.15)',
                border: '1px solid #e0e0e0',
                borderRadius: '4px',
                padding: '16px',
              },
              success: {
                iconTheme: {
                  primary: '#000',
                  secondary: '#fff',
                },
              },
              error: {
                iconTheme: {
                  primary: '#000',
                  secondary: '#fff',
                },
              },
            }}
          />
        </Providers>
      </body>
    </html>
  )
}