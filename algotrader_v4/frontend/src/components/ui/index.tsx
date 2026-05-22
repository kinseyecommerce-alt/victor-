import React from 'react'
import { clsx } from 'clsx'

export const Badge = ({
  children, variant = 'neutral'
}: { children: React.ReactNode; variant?: 'buy' | 'sell' | 'neutral' | 'warning' | 'paper' | 'live' }) => {
  const cls = {
    buy:     'bg-green-100 text-green-800',
    sell:    'bg-red-100 text-red-800',
    neutral: 'bg-slate-100 text-slate-700',
    warning: 'bg-amber-100 text-amber-800',
    paper:   'bg-amber-100 text-amber-800',
    live:    'bg-green-100 text-green-800',
  }[variant]
  return <span className={clsx('inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold', cls)}>{children}</span>
}

export const Btn = ({
  children, onClick, variant = 'primary', size = 'md', disabled, className, type = 'button'
}: {
  children: React.ReactNode
  onClick?: () => void
  variant?: 'primary' | 'buy' | 'sell' | 'danger' | 'ghost' | 'outline'
  size?: 'sm' | 'md' | 'lg'
  disabled?: boolean
  className?: string
  type?: 'button' | 'submit'
}) => {
  const variants = {
    primary: 'bg-indigo-600 hover:bg-indigo-700 text-white',
    buy:     'bg-green-600 hover:bg-green-700 text-white',
    sell:    'bg-red-600 hover:bg-red-700 text-white',
    danger:  'bg-red-600 hover:bg-red-700 text-white',
    ghost:   'bg-transparent hover:bg-slate-100 text-slate-700',
    outline: 'border border-slate-300 hover:bg-slate-50 text-slate-700',
  }[variant]
  const sizes = {
    sm: 'px-3 py-1 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base',
  }[size]
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        'rounded-lg font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-1',
        variants, sizes,
        disabled && 'opacity-50 cursor-not-allowed',
        className,
      )}
    >
      {children}
    </button>
  )
}

export const Card = ({ children, className }: { children: React.ReactNode; className?: string }) => (
  <div className={clsx('bg-white rounded-xl border border-slate-200 shadow-sm', className)}>
    {children}
  </div>
)

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={clsx(
        'w-full border border-slate-200 rounded-lg px-3 py-2 text-sm text-slate-900',
        'focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent',
        'placeholder:text-slate-400 bg-white',
        className,
      )}
      {...props}
    />
  )
)
Input.displayName = 'Input'

export const Select = React.forwardRef<HTMLSelectElement, React.SelectHTMLAttributes<HTMLSelectElement>>(
  ({ children, className, ...props }, ref) => (
    <select
      ref={ref}
      className={clsx(
        'w-full border border-slate-200 rounded-lg px-3 py-2 text-sm text-slate-900',
        'focus:outline-none focus:ring-2 focus:ring-indigo-500',
        'bg-white cursor-pointer',
        className,
      )}
      {...props}
    >
      {children}
    </select>
  )
)
Select.displayName = 'Select'

export const Pnl = ({ value }: { value: number }) => (
  <span className={clsx('font-mono text-sm font-medium', value >= 0 ? 'text-green-600' : 'text-red-600')}>
    {value >= 0 ? '+' : ''}₹{value.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
  </span>
)

export const Ltp = ({ value, prev }: { value: number; prev?: number }) => {
  const dir = prev === undefined ? '' : value > prev ? 'text-green-600' : value < prev ? 'text-red-600' : 'text-slate-900'
  return (
    <span className={clsx('font-mono font-semibold tabular-nums', dir || 'text-slate-900')}>
      ₹{value.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
    </span>
  )
}

export const StatusDot = ({ online }: { online: boolean }) => (
  <span className={clsx('inline-block w-2 h-2 rounded-full', online ? 'bg-green-500' : 'bg-red-500')} />
)

export const Modal = ({
  open, onClose, title, children
}: { open: boolean; onClose: () => void; title: string; children: React.ReactNode }) => {
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="relative bg-white rounded-2xl shadow-2xl p-6 w-full max-w-md mx-4">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">{title}</h2>
        {children}
      </div>
    </div>
  )
}
