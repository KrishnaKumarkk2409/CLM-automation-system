/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: ['localhost', '127.0.0.1'],
  },
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000',
  },
  output: 'standalone',
  // Fix webpack caching issues
  webpack: (config, { dev, isServer }) => {
    // Force cache invalidation during development
    if (dev) {
      config.cache = false;
    }
    return config;
  },
}

module.exports = nextConfig