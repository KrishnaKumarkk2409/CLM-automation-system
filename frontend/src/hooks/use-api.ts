import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for debugging
api.interceptors.request.use((config) => {
  console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
  return config
}, (error) => {
  console.warn('API Request Error:', error)
  return Promise.reject(error)
})

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Don't log ECONNREFUSED errors as they're common during development
    if (error.code !== 'ECONNREFUSED') {
      console.error('API Error:', error.response?.data || error.message)
    } else {
      console.warn('Backend connection failed - server may be starting up or unavailable')
    }
    return Promise.reject(error)
  }
)

// Types
export interface SystemStats {
  total_documents: number
  active_contracts: number
  total_chunks: number
  system_status: string
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  sources?: any[]
  timestamp?: Date
  message_id?: string
}

export interface ChatResponse {
  response: string
  sources: any[]
  conversation_id: string
  message_id: string
  timestamp: string
}

export interface DocumentSearchResult {
  documents: any[]
}

export interface AnalyticsData {
  overview: {
    total_documents: number
    active_contracts: number
    total_chunks: number
    system_status: string
  }
  distributions: {
    departments: Record<string, number>
    file_types: Record<string, number>
  }
  timeline: Array<{
    contract_id: string
    end_date: string
    days_until_expiry: number
    department: string
    status: string
  }>
  generated_at: string
}

export interface ExpiringContracts {
  expiring_contracts: any[]
  count: number
  days_threshold: number
  generated_at: string
}

export interface ConflictsData {
  conflicts: any[]
  count: number
  generated_at: string
}

export interface DocumentsListResponse {
  documents: any[]
  count: number
  limit: number
  offset: number
}

export interface SystemConfig {
  smtp_configured: boolean
  openai_configured: boolean
  supabase_configured: boolean
  documents_folder: string
  chunk_size: number
  similarity_threshold: number
  system_version: string
  timestamp: string
}

// Hooks
export function useSystemStats(options = {}) {
  return useQuery({
    queryKey: ['system-stats'],
    queryFn: async (): Promise<SystemStats> => {
      try {
        const response = await api.get('/stats')
        return response.data
      } catch (error) {
        console.warn('Failed to fetch system stats')
        // Return fallback data
        return {
          total_documents: 0,
          active_contracts: 0,
          total_chunks: 0,
          system_status: 'offline'
        }
      }
    },
    refetchInterval: 30000, // Refetch every 30 seconds
    ...options
  })
}

export function useHealthCheck() {
  return useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      const response = await api.get('/health')
      return response.data
    },
    refetchInterval: 60000, // Refetch every minute
  })
}

export function useSendMessage() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ message, conversationId }: { message: string; conversationId?: string }): Promise<ChatResponse> => {
      const response = await api.post('/chat', {
        message,
        conversation_id: conversationId,
      })
      return response.data
    },
    onSuccess: () => {
      // Invalidate any relevant queries
      queryClient.invalidateQueries({ queryKey: ['chat-history'] })
    },
  })
}

export function useUploadDocuments() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (files: File[]) => {
      const formData = new FormData()
      files.forEach((file) => {
        formData.append('files', file)
      })

      const response = await api.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      return response.data
    },
    onSuccess: () => {
      // Invalidate system stats to reflect new documents
      queryClient.invalidateQueries({ queryKey: ['system-stats'] })
    },
  })
}

export function useSearchDocuments() {
  return useMutation({
    mutationFn: async ({ query, limit = 10 }: { query: string; limit?: number }): Promise<DocumentSearchResult> => {
      const response = await api.post('/search', { query, limit })
      return response.data
    },
  })
}

export function useGenerateReport() {
  return useMutation({
    mutationFn: async (params: {
      email: string
      include_expiring?: boolean
      include_conflicts?: boolean
      include_analytics?: boolean
    }) => {
      const response = await api.post('/generate-report', params)
      return response.data
    },
  })
}

// Enhanced Analytics Hooks
export function useAnalyticsDashboard(options = {}) {
  return useQuery({
    queryKey: ['analytics-dashboard'],
    queryFn: async (): Promise<AnalyticsData> => {
      try {
        const response = await api.get('/analytics/dashboard')
        return response.data
      } catch (error) {
        console.warn('Failed to fetch analytics dashboard')
        // Return minimal fallback data
        return {
          overview: {
            total_documents: 0,
            active_contracts: 0,
            total_chunks: 0,
            system_status: 'offline'
          },
          distributions: {
            departments: {},
            file_types: {}
          },
          timeline: [],
          generated_at: new Date().toISOString()
        }
      }
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options
  })
}

export function useExpiringContracts(days: number = 30, options = {}) {
  return useQuery({
    queryKey: ['expiring-contracts', days],
    queryFn: async (): Promise<ExpiringContracts> => {
      try {
        const response = await api.get(`/analytics/contracts/expiring?days=${days}`)
        return response.data
      } catch (error) {
        console.warn('Failed to fetch expiring contracts')
        return {
          expiring_contracts: [],
          count: 0,
          days_threshold: days,
          generated_at: new Date().toISOString()
        }
      }
    },
    staleTime: 10 * 60 * 1000, // 10 minutes
    ...options
  })
}

export function useContractConflicts(options = {}) {
  return useQuery({
    queryKey: ['contract-conflicts'],
    queryFn: async (): Promise<ConflictsData> => {
      try {
        const response = await api.get('/analytics/conflicts')
        return response.data
      } catch (error) {
        console.warn('Failed to fetch contract conflicts')
        return {
          conflicts: [],
          count: 0,
          generated_at: new Date().toISOString()
        }
      }
    },
    staleTime: 15 * 60 * 1000, // 15 minutes
    ...options
  })
}

export function useGenerateAnalyticsReport() {
  return useMutation({
    mutationFn: async (email?: string) => {
      const response = await api.post('/analytics/report/generate', {
        email: email || null
      })
      return response.data
    },
  })
}

// Enhanced Document Management Hooks
export function useDocumentsList(params = { limit: 10, offset: 0 }, options = {}) {
  return useQuery({
    queryKey: ['documents-list', params],
    queryFn: async (): Promise<DocumentsListResponse> => {
      try {
        const { limit, offset, ...filters } = params
        const queryParams = new URLSearchParams({
          limit: limit.toString(),
          offset: offset.toString(),
          ...filters
        })
        
        const response = await api.get(`/documents?${queryParams}`)
        return response.data
      } catch (error) {
        console.warn('Failed to fetch documents list')
        return {
          documents: [],
          count: 0,
          limit: params.limit,
          offset: params.offset
        }
      }
    },
    staleTime: 2 * 60 * 1000, // 2 minutes
    ...options
  })
}

export function useAdvancedDocumentSearch() {
  return useMutation({
    mutationFn: async (params: {
      query: string
      limit?: number
      filters?: Record<string, any>
    }) => {
      const response = await api.post('/documents/search', params)
      return response.data
    },
  })
}

export function useProcessFolderDocuments() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async () => {
      const response = await api.post('/documents/process/folder')
      return response.data
    },
    onSuccess: () => {
      // Invalidate relevant queries
      queryClient.invalidateQueries({ queryKey: ['system-stats'] })
      queryClient.invalidateQueries({ queryKey: ['documents-list'] })
      queryClient.invalidateQueries({ queryKey: ['analytics-dashboard'] })
    },
  })
}

export function useProcessBatchDocuments() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async (documentIds: string[]) => {
      const response = await api.post('/documents/process/batch', documentIds)
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['system-stats'] })
      queryClient.invalidateQueries({ queryKey: ['documents-list'] })
    },
  })
}

// System Configuration Hooks
export function useSystemConfig() {
  return useQuery({
    queryKey: ['system-config'],
    queryFn: async (): Promise<SystemConfig> => {
      const response = await api.get('/system/config')
      return response.data
    },
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}

export function useUpdateEmailConfig() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async (emailConfig: {
      smtp_server?: string
      smtp_port?: number
      email_username?: string
      email_password?: string
    }) => {
      const response = await api.post('/system/config/email', emailConfig)
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['system-config'] })
    },
  })
}

// WebSocket hook for real-time chat
export function useWebSocket(conversationId: string) {
  const [socket, setSocket] = useState<WebSocket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [messages, setMessages] = useState<any[]>([])
  
  useEffect(() => {
    if (!conversationId) return
    
    const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'
    const ws = new WebSocket(`${WS_URL}/ws/chat/${conversationId}`)
    
    ws.onopen = () => {
      console.log('WebSocket connected')
      setIsConnected(true)
      setSocket(ws)
    }
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === 'chat_response') {
          setMessages(prev => [...prev, data])
        } else if (data.type === 'error') {
          console.error('WebSocket error:', data.error)
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error)
      }
    }
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setIsConnected(false)
    }
    
    ws.onclose = () => {
      console.log('WebSocket disconnected')
      setIsConnected(false)
      setSocket(null)
    }
    
    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close()
      }
    }
  }, [conversationId])
  
  const sendMessage = useCallback((message: string) => {
    if (socket && isConnected) {
      socket.send(JSON.stringify({
        type: 'chat',
        message: message
      }))
    }
  }, [socket, isConnected])
  
  const disconnect = useCallback(() => {
    if (socket) {
      socket.close()
    }
  }, [socket])
  
  return {
    socket,
    isConnected,
    messages,
    sendMessage,
    disconnect
  }
}
