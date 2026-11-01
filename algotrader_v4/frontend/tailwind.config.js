/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        buy:  { DEFAULT: '#16A34A', light: '#DCFCE7', dark: '#15803D' },
        sell: { DEFAULT: '#DC2626', light: '#FEE2E2', dark: '#B91C1C' },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      }
    },
  },
  plugins: [],
}
