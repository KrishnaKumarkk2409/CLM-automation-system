"use client"
import { useState } from 'react'
import { useGlobalSearch } from '../../hooks/use-api'

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [types, setTypes] = useState<string[]>(['documents', 'contracts', 'chunks'])
  const { mutateAsync, isPending, data, reset } = useGlobalSearch()

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return
    await mutateAsync({ query, limit: 20, search_types: types })
  }

  const toggleType = (t: string) => {
    setTypes((prev) => (prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]))
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-5xl mx-auto p-6">
        <h2 className="text-xl font-semibold mb-4">Global Search</h2>
        <form onSubmit={onSubmit} className="bg-white rounded-md border p-4 flex flex-col gap-3">
          <div className="flex gap-2">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search documents, contracts, and chunks..."
              className="flex-1 border rounded-md px-3 py-2"
            />
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-md disabled:opacity-60"
              disabled={isPending}
            >
              {isPending ? 'Searching...' : 'Search'}
            </button>
          </div>
          <div className="flex items-center gap-4 text-sm">
            {['documents', 'contracts', 'chunks'].map((t) => (
              <label key={t} className="flex items-center gap-2">
                <input type="checkbox" checked={types.includes(t)} onChange={() => toggleType(t)} />
                <span className="capitalize">{t}</span>
              </label>
            ))}
          </div>
        </form>

        <div className="mt-6">
          {data?.results?.length ? (
            <div className="bg-white rounded-md border divide-y">
              {data.results.map((r) => (
                <div key={`${r.type}:${r.id}`} className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="font-medium">{r.title}</div>
                    <span className="text-xs px-2 py-1 rounded bg-gray-100 border">{r.type}</span>
                  </div>
                  <p className="text-sm text-gray-600 mt-2 whitespace-pre-wrap">{r.content_preview}</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500">Enter a query to search across all connected data.</p>
          )}
        </div>
      </div>
    </div>
  )
}


