import type { ReactNode } from 'react'

export function PageHeader({eyebrow, title, children}: {eyebrow: string, title: string, children: ReactNode}) {
  return <header className="page-header"><span className="eyebrow">{eyebrow}</span><h1>{title}</h1><p>{children}</p></header>
}

export function Card({children, className = ''}: {children: ReactNode, className?: string}) {
  return <section className={`card ${className}`}>{children}</section>
}

export function Loading() { return <div className="loading"><span/><span/><span/></div> }
export function ErrorState({error}: {error: Error}) { return <div className="error">{error.message}</div> }

export function Metric({label, value, note}: {label: string, value: string, note: string}) {
  return <Card className="metric"><span>{label}</span><strong>{value}</strong><small>{note}</small></Card>
}

export const fmt = (value: number, digits = 3) => value.toFixed(digits)
