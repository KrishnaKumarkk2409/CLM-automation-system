/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Design system colors using HSL values
        primary: {
          DEFAULT: 'hsl(214, 88%, 27%)',
          hover: 'hsl(214, 88%, 35%)',
          50: 'hsl(214, 88%, 95%)',
          100: 'hsl(214, 88%, 90%)',
          200: 'hsl(214, 88%, 80%)',
          300: 'hsl(214, 88%, 70%)',
          400: 'hsl(214, 88%, 60%)',
          500: 'hsl(214, 88%, 50%)',
          600: 'hsl(214, 88%, 27%)', // Corporate Blue
          700: 'hsl(214, 88%, 20%)',
          800: 'hsl(214, 88%, 15%)',
          900: 'hsl(214, 88%, 10%)',
        },
        background: 'hsl(210, 20%, 98%)', // Light gray 248 250 252
        foreground: 'hsl(203, 23%, 30%)', // Dark blue-gray
        success: 'hsl(142, 76%, 36%)', // Green
        warning: 'hsl(38, 92%, 50%)', // Orange
        destructive: 'hsl(0, 84%, 60%)', // Red
        muted: 'hsl(210, 40%, 98%)', // Light gray
        // Keep secondary colors for backward compatibility
        secondary: {
          50: 'hsl(210, 20%, 98%)',
          100: 'hsl(210, 20%, 96%)',
          200: 'hsl(210, 16%, 93%)',
          300: 'hsl(210, 14%, 89%)',
          400: 'hsl(203, 15%, 65%)',
          500: 'hsl(203, 18%, 47%)',
          600: 'hsl(203, 23%, 30%)', // Foreground color
          700: 'hsl(205, 30%, 27%)',
          800: 'hsl(205, 30%, 15%)',
          900: 'hsl(220, 40%, 10%)',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Monaco', 'Consolas', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'spin-slow': 'spin 3s linear infinite',
        'bounce-light': 'bounceLight 1s infinite',
        'pulse-soft': 'pulseSoft 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        bounceLight: {
          '0%, 20%, 50%, 80%, 100%': { transform: 'translateY(0)' },
          '40%': { transform: 'translateY(-10px)' },
          '60%': { transform: 'translateY(-5px)' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '0.8' },
          '50%': { opacity: '1' },
        },
      },
      boxShadow: {
        'glass': '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
        'glass-hover': '0 12px 40px 0 rgba(31, 38, 135, 0.45)',
        'soft': '0 2px 15px -3px rgba(0, 0, 0, 0.07), 0 10px 20px -2px rgba(0, 0, 0, 0.04)',
        'soft-lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
      },
      backdropBlur: {
        xs: '2px',
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem',
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}