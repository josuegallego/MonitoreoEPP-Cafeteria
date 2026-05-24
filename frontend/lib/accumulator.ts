import { Inspection, EPP_CLASSES, EppClass } from './types'

const KEY = 'epp_turno_v2'

export interface TurnoStore {
  date: string
  inspections: Inspection[]
}

export function getTurno(): TurnoStore {
  if (typeof window === 'undefined') return fresh()
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return fresh()
    const d: TurnoStore = JSON.parse(raw)
    if (d.date !== today()) return fresh()
    return d
  } catch { return fresh() }
}

export function saveTurno(t: TurnoStore) {
  if (typeof window === 'undefined') return
  localStorage.setItem(KEY, JSON.stringify(t))
}

export function resetTurno(): TurnoStore {
  const f = fresh()
  if (typeof window !== 'undefined') localStorage.setItem(KEY, JSON.stringify(f))
  return f
}

export function addInspection(t: TurnoStore, ins: Inspection): TurnoStore {
  // Omite annotated_image al persistir (base64 de ~300KB por inspección agota localStorage)
  const lean: Inspection = { ...ins, result: { ...ins.result, annotated_image: '' } }
  const u = { ...t, inspections: [...t.inspections, lean] }
  saveTurno(u)
  return u
}

function fresh(): TurnoStore { return { date: today(), inspections: [] } }
function today() { return new Date().toLocaleDateString('es-CO') }

export function computeStats(t: TurnoStore) {
  const ins = t.inspections
  if (!ins.length) return null

  const globalPct = Math.round(ins.reduce((s, i) => s + i.result.global_compliance, 0) / ins.length)
  const violations = ins.filter(i => i.result.persons.some(p => p.violations.length > 0)).length

  const breakdown = Object.fromEntries(EPP_CLASSES.map(epp => {
    const total    = ins.length
    const detected = ins.filter(i => i.result.persons.some(p => p.detected.includes(epp))).length
    return [epp, { total, detected, pct: total > 0 ? Math.round(detected / total * 100) : 0 }]
  })) as Record<EppClass, { total: number; detected: number; pct: number }>

  const worst = EPP_CLASSES.reduce((a, b) => breakdown[a].pct <= breakdown[b].pct ? a : b)

  const hourMap: Record<string, number> = {}
  for (const i of ins) {
    if (i.result.persons.some(p => p.violations.length > 0)) {
      hourMap[i.time] = (hourMap[i.time] || 0) + 1
    }
  }
  const peakHour = Object.keys(hourMap).length
    ? Object.entries(hourMap).sort((a, b) => b[1] - a[1])[0][0]
    : '--'

  return { globalPct, violations, breakdown, worst, peakHour, total: ins.length }
}
