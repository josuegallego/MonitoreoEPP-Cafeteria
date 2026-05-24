'use client'
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, ReferenceLine, Area, AreaChart } from 'recharts'
import { Inspection } from '@/lib/types'

export default function TrendChart({ inspections }: { inspections: Inspection[] }) {
  const data = inspections.map(i => ({
    time: i.time,
    pct: i.result.global_compliance,
  }))

  if (!data.length) return (
    <div style={{ height: 120, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text3)', fontSize: 12 }}>
      Sin datos aún
    </div>
  )

  return (
    <ResponsiveContainer width="100%" height={120}>
      <AreaChart data={data} margin={{ top: 8, right: 4, bottom: 0, left: -24 }}>
        <defs>
          <linearGradient id="tealGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#00e5b0" stopOpacity={0.15} />
            <stop offset="95%" stopColor="#00e5b0" stopOpacity={0} />
          </linearGradient>
        </defs>
        <XAxis dataKey="time" tick={{ fill: 'var(--text3)', fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis domain={[0, 100]} tick={{ fill: 'var(--text3)', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} />
        <Tooltip
          contentStyle={{ background: '#1a2236', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, fontSize: 12 }}
          labelStyle={{ color: 'var(--text2)' }} itemStyle={{ color: '#00e5b0' }}
          formatter={(v: number) => [`${v}%`, 'Cumplimiento']}
        />
        <ReferenceLine y={70} stroke="rgba(255,69,96,0.25)" strokeDasharray="3 3" />
        <Area type="monotone" dataKey="pct" stroke="#00e5b0" strokeWidth={2}
          fill="url(#tealGrad)" dot={{ r: 3, fill: '#00e5b0', strokeWidth: 0 }}
          activeDot={{ r: 5, fill: '#00e5b0', stroke: 'rgba(0,229,176,0.3)', strokeWidth: 3 }}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
