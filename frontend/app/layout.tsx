import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'EPP Monitor · Cafetería UAO',
  description: 'Sistema de monitoreo de higiene y bioseguridad',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  )
}
