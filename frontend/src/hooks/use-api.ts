import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { supabase } from '@/lib/supabase'
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
  // Attach Supabase JWT if available
  const accessToken = (supabase as any)?.auth?.getSession ? undefined : undefined
  return config
})

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message)
    throw error
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
  user_id?: string
}

export interface DocumentSearchResult {
  documents: any[]
}

export interface AnalyticsData {
  contract_timeline: any[]
  department_distribution: Record<string, number>
  expiring_contracts: number
  total_value: number
}

// Hooks
export function useSystemStats() {
  return useQuery({
    queryKey: ['system-stats'],
    queryFn: async (): Promise<SystemStats> => {
      const response = await api.get('/stats')
      return response.data
    },
    refetchInterval: 30000, // Refetch every 30 seconds
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
      // Ensure auth header is present for this call
      const { data: sessionData } = await supabase.auth.getSession()
      const token = sessionData.session?.access_token
      const response = await api.post('/chat', {
        message,
        conversation_id: conversationId,
      }, { headers: token ? { Authorization: `Bearer ${token}` } : undefined })
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
      // Invalidate system stats and document lists to reflect new documents
      queryClient.invalidateQueries({ queryKey: ['system-stats'] })
      queryClient.invalidateQueries({ queryKey: ['frontend-documents-list'] })
      queryClient.invalidateQueries({ queryKey: ['documents-list'] })
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

export interface GlobalSearchParams {
  query: string
  limit?: number
  search_types?: string[]
}

export interface GlobalSearchItem {
  id: string
  title: string
  type: 'document' | 'contract' | 'chunk'
  content_preview: string
  score: number
  metadata: Record<string, any>
}

export interface GlobalSearchResponse {
  results: GlobalSearchItem[]
}

export function useGlobalSearch() {
  return useMutation({
    mutationFn: async ({ query, limit = 20, search_types = ['documents', 'contracts', 'chunks'] }: GlobalSearchParams): Promise<GlobalSearchResponse> => {
      const response = await api.post('/global-search', { query, limit, search_types })
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

export function useAnalytics() {
  return useQuery({
    queryKey: ['analytics'],
    queryFn: async (): Promise<AnalyticsData> => {
      const response = await api.get('/analytics')
      return response.data
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

// Documents and Contracts hooks
export function useDocumentsList() {
  return useQuery({
    queryKey: ['documents-list'],
    queryFn: async () => {
      const response = await api.get('/documents')
      return response.data
    },
    staleTime: 2 * 60 * 1000, // 2 minutes
  })
}

export function useContractsList() {
  return useQuery({
    queryKey: ['contracts-list'],
    queryFn: async () => {
      const response = await api.get('/contracts')
      return response.data
    },
    staleTime: 2 * 60 * 1000, // 2 minutes
  })
}

export function useFrontendDocumentsList() {
  return useQuery({
    queryKey: ['frontend-documents-list'],
    queryFn: async () => {
      const response = await api.get('/frontend-documents')
      return response.data
    },
    staleTime: 2 * 60 * 1000, // 2 minutes
  })
}

// API Key Management hooks
export function useAPIKeys() {
  return useQuery({
    queryKey: ['api-keys'],
    queryFn: async () => {
      const response = await api.get('/api-keys')
      return response.data
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

export function useCreateAPIKey() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (keyData: {
      name: string
      description?: string
      expires_at?: string
    }) => {
      const response = await api.post('/api-keys', keyData)
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['api-keys'] })
    },
  })
}

export function useDeleteAPIKey() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (keyId: string) => {
      const response = await api.delete(`/api-keys/${keyId}`)
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['api-keys'] })
    },
  })
}

// WebSocket hook for real-time chat
export function useWebSocket(conversationId: string) {
  // This would implement WebSocket connection
  // For now, returning a placeholder
  return {
    socket: null,
    isConnected: false,
    sendMessage: (message: string) => {
      console.log('WebSocket send:', message)
    },
    disconnect: () => {
      console.log('WebSocket disconnect')
    },
  }
}

// Chat history types
export interface ChatHistoryResponse {
  messages: {
    conversation_id: string
    role: 'user' | 'assistant'
    content: string
    created_at: string
    id?: string
  }[]
  pagination: {
    next_cursor: string | null
    has_more: boolean
    limit: number
  }
}

// Chat history with pagination support
export function useChatHistory(conversationId?: string, limit: number = 50) {
  return useQuery({
    queryKey: ['chat-history', conversationId || 'all', limit],
    queryFn: async (): Promise<ChatHistoryResponse> => {
      const { data: sessionData } = await supabase.auth.getSession()
      const token = sessionData.session?.access_token
      const params: Record<string, any> = { limit }
      if (conversationId) {
        params.conversation_id = conversationId
      }
      const response = await api.get('/chat-history', {
        params,
        headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      })
      return response.data
    },
    staleTime: 30 * 1000,
    enabled: true,
  })
}

// Hook for paginated chat history (for loading more messages)
export function useChatHistoryPaginated() {
  return useMutation({
    mutationFn: async ({
      conversationId,
      cursor,
      limit = 50
    }: {
      conversationId?: string
      cursor: string
      limit?: number
    }): Promise<ChatHistoryResponse> => {
      const { data: sessionData } = await supabase.auth.getSession()
      const token = sessionData.session?.access_token
      const params: Record<string, any> = { limit, cursor }
      if (conversationId) {
        params.conversation_id = conversationId
      }
      const response = await api.get('/chat-history', {
        params,
        headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      })
      return response.data
    },
  })
}
