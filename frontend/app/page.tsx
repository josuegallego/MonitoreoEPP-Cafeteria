'use client'
import { useState, useEffect, useRef, useCallback } from 'react'
import dynamic from 'next/dynamic'
import { EPP_CLASSES, EPP_META, EppClass, DetectionResult, Inspection, Person } from '@/lib/types'
import { getTurno, addInspection, resetTurno, computeStats, TurnoStore } from '@/lib/accumulator'

const TrendChart = dynamic(() => import('@/components/TrendChart'), { ssr: false })

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000'
const SIM_IMAGES = Array.from({ length: 39 }, (_, i) => `/test/prueba${i + 1}.jpg`)
  .map(v => ({ v, r: Math.random() })).sort((a, b) => a.r - b.r).map(x => x.v)

// ── Stat card ─────────────────────────────────────────────────
function StatCard({ label, value, sub, color = 'var(--text)' }: {
  label: string; value: string; sub?: string; color?: string
}) {
  return (
    <div className="card" style={{ padding: '14px 16px' }}>
      <div className="label" style={{ marginBottom: 8 }}>{label}</div>
      <div className="mono" style={{ fontSize: 28, fontWeight: 600, color, lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 4 }}>{sub}</div>}
    </div>
  )
}

// ── EPP tag ──────────────────────────────────────────────────
function EppTag({ epp, present }: { epp: EppClass; present: boolean }) {
  const m = EPP_META[epp]
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      padding: '3px 9px', borderRadius: 6, fontSize: 11, fontWeight: 500,
      background: present ? `${m.color}18` : 'rgba(255,69,96,0.12)',
      border: `1px solid ${present ? m.color + '33' : 'rgba(255,69,96,0.25)'}`,
      color: present ? m.color : 'var(--red)',
    }}>
      {present ? '+' : 'X'} {m.label}
    </span>
  )
}

// ── Person card ──────────────────────────────────────────────
function PersonCard({ person, idx }: { person: Person; idx: number }) {
  const ok = person.violations.length === 0
  const c  = person.compliance >= 90 ? 'var(--teal)' : person.compliance >= 60 ? 'var(--amber)' : 'var(--red)'
  return (
    <div className="fadeup" style={{
      background: 'var(--bg3)', border: `1px solid ${ok ? 'rgba(0,229,176,0.15)' : 'rgba(255,69,96,0.15)'}`,
      borderRadius: 10, padding: '12px 14px',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 28, height: 28, borderRadius: '50%', background: ok ? 'var(--teal-dim)' : 'var(--red-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 700, color: ok ? 'var(--teal)' : 'var(--red)' }}>
            P
          </div>
          <div>
            <div style={{ fontSize: 13, fontWeight: 600 }}>Persona {idx + 1}</div>
            <div style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'var(--mono)' }}>
              conf. {Math.round(person.conf * 100)}%
            </div>
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div className="mono" style={{ fontSize: 22, fontWeight: 600, color: c }}>{person.compliance}%</div>
          <div style={{ fontSize: 10, color: ok ? 'var(--teal)' : 'var(--red)' }}>{ok ? 'cumple' : 'incumple'}</div>
        </div>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
        {EPP_CLASSES.map(epp => (
          <EppTag key={epp} epp={epp} present={person.detected.includes(epp)} />
        ))}
      </div>
      {/* Prob bars */}
      <div style={{ marginTop: 10, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '5px 12px' }}>
        {EPP_CLASSES.map(epp => (
          <div key={epp}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
              <span style={{ fontSize: 10, color: 'var(--text3)', textTransform: 'capitalize' }}>{epp}</span>
              <span className="mono" style={{ fontSize: 10, color: EPP_META[epp].color }}>{Math.round((person.probs[epp] ?? 0) * 100)}%</span>
            </div>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${(person.probs[epp] ?? 0) * 100}%`, background: EPP_META[epp].color }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── PPE Breakdown ────────────────────────────────────────────
function PPEBreakdown({ breakdown }: { breakdown: Record<EppClass, { pct: number }> | null }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {EPP_CLASSES.map(epp => {
        const pct = breakdown?.[epp]?.pct ?? null
        const m   = EPP_META[epp]
        const c   = pct === null ? 'var(--text3)' : pct < 70 ? 'var(--red)' : pct < 90 ? 'var(--amber)' : 'var(--teal)'
        return (
          <div key={epp}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5, alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                <span style={{ fontSize: 14 }}>{m.icon}</span>
                <span style={{ fontSize: 13 }}>{m.label}</span>
              </div>
              <span className="mono" style={{ fontSize: 13, fontWeight: 600, color: c }}>
                {pct !== null ? `${pct}%` : '--'}
              </span>
            </div>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${pct ?? 0}%`, background: m.color, opacity: pct === null ? 0 : 1 }} />
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ── Main Dashboard ───────────────────────────────────────────
export default function Dashboard() {
  const [turno, setTurno]             = useState<TurnoStore | null>(null)
  const [annotated, setAnnotated]     = useState<string | null>(null)  // base64
  const [lastResult, setLastResult]   = useState<DetectionResult | null>(null)
  const [loading, setLoading]         = useState(false)
  const [backendOk, setBackendOk]     = useState<boolean | null>(null)
  const [backendMsg, setBackendMsg]   = useState('Conectando...')
  const [simRunning, setSimRunning]   = useState(false)
  const [simMode, setSimMode]         = useState<'demo'|'real'>('demo')
  const [countdown, setCountdown]     = useState<number | null>(null)
  const [clock, setClock]             = useState('')
  const [error, setError]             = useState<string | null>(null)
  const simRef   = useRef<ReturnType<typeof setInterval> | null>(null)
  const cdRef    = useRef<ReturnType<typeof setInterval> | null>(null)
  const simIdx   = useRef(0)
  const fileRef  = useRef<HTMLInputElement>(null)
  const alarmRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    alarmRef.current = new Audio('/alarma.mp3')
    alarmRef.current.volume = 0.85
  }, [])

  useEffect(() => { setTurno(getTurno()) }, [])
  useEffect(() => {
    const t = setInterval(() => setClock(new Date().toLocaleTimeString('es-CO', { hour: '2-digit', minute: '2-digit', second: '2-digit' })), 1000)
    return () => clearInterval(t)
  }, [])

  // Health check con timeout y reintentos (Render free tier tarda ~30-60s en despertar)
  useEffect(() => {
    let cancelled = false
    let attempt = 0
    const check = async () => {
      if (cancelled) return
      attempt++
      if (attempt > 1) setBackendMsg(`Iniciando backend... (${attempt}s)`)
      try {
        const ctrl = new AbortController()
        const tid  = setTimeout(() => ctrl.abort(), 8000)
        const r    = await fetch(`${BACKEND}/health`, { signal: ctrl.signal })
        clearTimeout(tid)
        if (!cancelled) { setBackendOk(r.ok); setBackendMsg(r.ok ? 'Backend conectado' : 'Backend desconectado') }
      } catch {
        if (!cancelled) { setTimeout(check, 3000) }
      }
    }
    check()
    return () => { cancelled = true }
  }, [])

  const stats = turno ? computeStats(turno) : null

  // ── Send image to backend ────────────────────────────────
  const sendImage = useCallback(async (file: File | Blob) => {
    setLoading(true); setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${BACKEND}/detect`, { method: 'POST', body: form })
      if (!res.ok) { const t = await res.text(); throw new Error(t) }
      const data: DetectionResult = await res.json()
      setAnnotated(`data:image/jpeg;base64,${data.annotated_image}`)
      setLastResult(data)
      const incumple = data.persons.filter(p => p.violations.length > 0).length
      const mayoriaIncumple = incumple >= Math.ceil(data.persons.length / 2)
      if (mayoriaIncumple && alarmRef.current) {
        alarmRef.current.currentTime = 0
        alarmRef.current.play().catch(() => {})
      }
      const time = new Date().toLocaleTimeString('es-CO', { hour: '2-digit', minute: '2-digit' })
      const ins: Inspection = { id: Date.now().toString(), time, timestamp: Date.now(), result: data }
      setTurno(prev => prev ? addInspection(prev, ins) : prev)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally { setLoading(false) }
  }, [])

  const handleFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (!f) return
    setAnnotated(URL.createObjectURL(f))
    sendImage(f)
    e.target.value = ''
  }, [sendImage])

  // ── Simulation ───────────────────────────────────────────
  const startSim = useCallback(() => {
    const ms = simMode === 'demo' ? 12_000 : 5 * 60 * 1000
    setSimRunning(true); setCountdown(ms / 1000)

    const run = async () => {
      const url = SIM_IMAGES[simIdx.current % SIM_IMAGES.length]; simIdx.current++
      try {
        const res  = await fetch(url); const blob = await res.blob(); await sendImage(blob)
      } catch { /* skip */ }
      setCountdown(ms / 1000)
    }
    run()
    simRef.current = setInterval(run, ms)
    cdRef.current  = setInterval(() => setCountdown(c => c !== null ? Math.max(0, c - 1) : null), 1000)
  }, [simMode, sendImage])

  const stopSim = useCallback(() => {
    setSimRunning(false); setCountdown(null)
    if (simRef.current) clearInterval(simRef.current)
    if (cdRef.current)  clearInterval(cdRef.current)
  }, [])
  useEffect(() => () => stopSim(), [stopSim])

  const globalColor = !stats ? 'var(--text2)'
    : stats.globalPct >= 90 ? 'var(--teal)'
    : stats.globalPct >= 70 ? 'var(--amber)' : 'var(--red)'

  const statusLabel = !stats ? 'SIN DATOS'
    : stats.globalPct >= 90 ? 'EXCELENTE'
    : stats.globalPct >= 70 ? 'ACEPTABLE' : 'CRÍTICO'

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>

      {/* ── Topbar ── */}
      <header style={{ background: 'var(--bg2)', borderBottom: '1px solid var(--border)', padding: '10px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', position: 'sticky', top: 0, zIndex: 100, backdropFilter: 'blur(12px)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div className="live-dot" />
          <span style={{ fontSize: 13, fontWeight: 700, letterSpacing: '0.04em', color: 'var(--text)', display: 'flex', alignItems: 'center', gap: 6 }}>
            SISTEMA EPP · CAFETERÍA
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/uao.png" alt="UAO" style={{ height: 22, width: 'auto', objectFit: 'contain', display: 'block' }} />
          </span>
          <span className="badge badge-blue" style={{ marginLeft: 4 }}>PDI 2026</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 6, height: 6, borderRadius: '50%', background: backendOk === true ? 'var(--teal)' : backendOk === false ? 'var(--red)' : 'var(--amber)' }} />
            <span style={{ fontSize: 11, color: 'var(--text2)' }}>{backendMsg}</span>
          </div>
          <span className="mono" style={{ fontSize: 12, color: 'var(--text2)' }}>{clock}</span>
          <span className="badge badge-teal">
            {new Date().toLocaleDateString('es-CO', { day: 'numeric', month: 'short' })}
          </span>
        </div>
      </header>

      <main style={{ flex: 1, padding: '16px 20px', display: 'grid', gridTemplateColumns: '1.25fr 1fr', gap: 16, alignItems: 'start' }}>

        {/* ══ LEFT ══ */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

          {/* Camera feed */}
          <div className="card">
            <div className="card-header">
              <div>
                <div className="label">Vista de cámara — Zona de cocina</div>
                {countdown !== null && (
                  <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 2, fontFamily: 'var(--mono)' }}>
                    próxima captura: {countdown}s
                  </div>
                )}
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="btn btn-ghost" style={{ fontSize: 11, padding: '5px 10px' }}
                  onClick={() => fileRef.current?.click()} disabled={loading}>
                  ↑ Subir imagen
                </button>
                <input ref={fileRef} type="file" accept="image/*" onChange={handleFile} style={{ display: 'none' }} />
              </div>
            </div>

            <div style={{ position: 'relative', height: 480, background: '#05080f', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }} className="scanlines">
              {/* Grid overlay */}
              <div style={{ position: 'absolute', inset: 0, backgroundImage: 'linear-gradient(rgba(0,229,176,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,229,176,0.03) 1px, transparent 1px)', backgroundSize: '40px 40px', pointerEvents: 'none' }} />

              {loading && (
                <div style={{ position: 'absolute', inset: 0, background: 'rgba(5,8,15,0.85)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12, zIndex: 10 }}>
                  <div className="spinner" />
                  <span style={{ fontSize: 12, color: 'var(--teal)', fontFamily: 'var(--mono)' }}>Analizando pipeline EPP...</span>
                  <div style={{ display: 'flex', gap: 6 }}>
                    {['Personas', 'Recortes', 'CNN EPP'].map((s, i) => (
                      <span key={s} className="badge badge-blue" style={{ opacity: 0.6 + i * 0.1 }}>{s}</span>
                    ))}
                  </div>
                </div>
              )}

              {annotated ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={annotated} alt="Detección EPP" style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block', padding: 8 }} />
              ) : (
                <label style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10, padding: 48 }}>
                  <div style={{ width: 56, height: 56, borderRadius: 12, background: 'rgba(0,229,176,0.08)', border: '1px solid rgba(0,229,176,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 22, color: 'var(--teal)', fontWeight: 700 }}>+</div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: 14, color: 'var(--text2)', fontWeight: 500 }}>Subir imagen para analizar</div>
                    <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 4 }}>o activar la simulación automática</div>
                  </div>
                  <input type="file" accept="image/*" onChange={handleFile} style={{ display: 'none' }} />
                </label>
              )}
            </div>

            {error && (
              <div style={{ padding: '8px 14px', background: 'rgba(255,69,96,0.08)', borderTop: '1px solid rgba(255,69,96,0.15)', fontSize: 12, color: 'var(--red)' }}>
                ! {error}
              </div>
            )}
          </div>

          {/* Persons grid */}
          {lastResult && lastResult.persons.length > 0 && (
            <div>
              <div className="label" style={{ marginBottom: 10, paddingLeft: 2 }}>
                PERSONAS DETECTADAS ({lastResult.total_persons})
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: lastResult.persons.length > 1 ? '1fr 1fr' : '1fr', gap: 10 }}>
                {lastResult.persons.map((p, i) => <PersonCard key={p.id} person={p} idx={i} />)}
              </div>
            </div>
          )}

          {/* Simulation controls */}
          <div className="card">
            <div className="card-header">
              <div className="label">Simulación automática</div>
              {simRunning && countdown !== null && (
                <span className="mono" style={{ fontSize: 11, color: 'var(--text3)' }}>
                  próx. captura: {countdown}s
                </span>
              )}
            </div>
            <div style={{ padding: '12px 14px', display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', borderRadius: 8, overflow: 'hidden', border: '1px solid var(--border2)' }}>
                {(['demo', 'real'] as const).map((m, i) => (
                  <button key={m} disabled={simRunning}
                    onClick={() => setSimMode(m)}
                    style={{ padding: '6px 12px', fontSize: 11, fontWeight: 500, cursor: simRunning ? 'not-allowed' : 'pointer', background: simMode === m ? 'var(--teal-dim)' : 'transparent', color: simMode === m ? 'var(--teal)' : 'var(--text3)', border: 'none', borderRight: i === 0 ? '1px solid var(--border2)' : 'none', fontFamily: 'var(--sans)', transition: 'all 0.15s' }}>
                    {m === 'demo' ? 'Demo 12s' : 'Real 5min'}
                  </button>
                ))}
              </div>
              {simRunning ? (
                <button className="btn btn-danger" onClick={stopSim}>Detener</button>
              ) : (
                <button className="btn btn-primary" onClick={startSim} disabled={backendOk !== true || loading}>
                  Iniciar simulación
                </button>
              )}
              <button className="btn btn-ghost" onClick={() => { setTurno(resetTurno()); setAnnotated(null); setLastResult(null) }}>
                Resetear turno
              </button>
            </div>
          </div>

          {/* Inspection log */}
          <div className="card">
            <div className="card-header">
              <span className="label">Registro de inspecciones</span>
              <span className="mono" style={{ fontSize: 11, color: 'var(--text3)' }}>{turno?.inspections.length ?? 0} registros</span>
            </div>
            <div style={{ maxHeight: 200, overflowY: 'auto' }}>
              {!turno?.inspections.length ? (
                <div style={{ padding: '20px 16px', fontSize: 12, color: 'var(--text3)', textAlign: 'center' }}>Sin inspecciones aún</div>
              ) : [...turno.inspections].reverse().map(ins => {
                const ok = ins.result.persons.every(p => p.violations.length === 0)
                const c  = ins.result.global_compliance >= 90 ? 'var(--teal)' : ins.result.global_compliance >= 70 ? 'var(--amber)' : 'var(--red)'
                return (
                  <div key={ins.id} style={{ padding: '9px 16px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                      <span className={`badge ${ok ? 'badge-teal' : 'badge-red'}`}>{ok ? 'OK' : '!'}</span>
                      <span className="mono" style={{ fontSize: 12, color: 'var(--text)' }}>{ins.time}</span>
                      <span style={{ fontSize: 11, color: 'var(--text3)' }}>{ins.result.total_persons} persona(s)</span>
                    </div>
                    <span className="mono" style={{ fontSize: 14, fontWeight: 600, color: c }}>{ins.result.global_compliance}%</span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        {/* ══ RIGHT ══ */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

          {/* Big compliance card */}
          <div className="card" style={{ padding: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20 }}>
              <div>
                <div className="label" style={{ marginBottom: 4 }}>Resumen de cumplimiento</div>
                <div style={{ fontSize: 12, color: 'var(--text3)' }}>
                  Hoy, {new Date().toLocaleDateString('es-CO', { weekday: 'long', day: 'numeric', month: 'long' })}
                </div>
              </div>
              <span style={{
                fontSize: 11, fontWeight: 700, padding: '4px 10px', borderRadius: 6, letterSpacing: '0.05em',
                background: !stats ? 'var(--bg3)' : stats.globalPct >= 90 ? 'var(--teal-dim)' : stats.globalPct >= 70 ? 'var(--amber-dim)' : 'var(--red-dim)',
                color: globalColor, border: `1px solid ${!stats ? 'var(--border)' : globalColor + '44'}`,
              }}>{statusLabel}</span>
            </div>

            <div style={{ marginBottom: 20 }}>
              <span className="mono" style={{ fontSize: 60, fontWeight: 700, color: globalColor, lineHeight: 1 }}>
                {stats ? `${stats.globalPct}%` : '--%'}
              </span>
              <div style={{ fontSize: 11, color: 'var(--text2)', marginTop: 6, letterSpacing: '0.06em', fontWeight: 500 }}>
                CUMPLIMIENTO GENERAL DE EPP
              </div>
              {stats && (
                <div style={{ fontSize: 12, color: 'var(--text3)', marginTop: 3 }}>
                  {stats.total} inspecciones · {stats.violations} con incumplimientos
                </div>
              )}
            </div>

            <div style={{ height: 1, background: 'var(--border)', margin: '16px 0' }} />
            <PPEBreakdown breakdown={stats?.breakdown ?? null} />
          </div>

          {/* Mini stats */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <StatCard label="Hora crítica" value={stats?.peakHour ?? '--'}
              sub="mayor incumplimiento" color="var(--amber)" />
            <StatCard label="EPP crítico" value={stats?.worst ?? '--'}
              sub={stats ? `${stats.breakdown[stats.worst as EppClass]?.pct}% cumplimiento` : '--'}
              color="var(--red)" />
          </div>

          {/* Trend */}
          <div className="card" style={{ padding: '14px' }}>
            <div className="label" style={{ marginBottom: 10 }}>Tendencia del turno</div>
            <TrendChart inspections={turno?.inspections ?? []} />
          </div>

          {/* Alert */}
          <div className="card">
            <div className="card-header">
              <span className="label">Alertas activas</span>
            </div>
            <div style={{ padding: '12px 14px' }}>
              {stats && stats.breakdown[stats.worst as EppClass]?.pct < 70 ? (
                <div style={{ display: 'flex', gap: 10, padding: '10px 12px', background: 'rgba(255,69,96,0.07)', border: '1px solid rgba(255,69,96,0.18)', borderRadius: 8 }}>
                  <span style={{ fontSize: 18, fontWeight: 700, color: 'var(--red)' }}>!</span>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--red)', marginBottom: 3 }}>
                      {stats.worst} — EPP más incumplido en el turno
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text3)' }}>
                      Cumplimiento: {stats.breakdown[stats.worst as EppClass]?.pct}% · umbral: 70%
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 0' }}>
                  <span style={{ fontSize: 12, color: 'var(--text3)' }}>Sin alertas activas en este turno</span>
                </div>
              )}
            </div>
          </div>

          {/* Download report */}
          <button className="btn btn-primary" disabled={!stats}
            style={{ width: '100%', justifyContent: 'center', padding: '13px', fontSize: 13, borderRadius: 10, fontWeight: 600, letterSpacing: '0.03em' }}
            onClick={() => {
              if (!turno || !stats) return
              const lines = [
                'REPORTE DE TURNO — EPP CAFETERÍA UAO',
                `Fecha: ${turno.date}`,
                `Cumplimiento global: ${stats.globalPct}%`,
                `Total inspecciones: ${stats.total}`,
                `Inspecciones con incumplimientos: ${stats.violations}`,
                `Hora crítica: ${stats.peakHour}`,
                `EPP más crítico: ${stats.worst}`,
                '',
                'PPE BREAKDOWN:',
                ...EPP_CLASSES.map(e => `  ${e}: ${stats.breakdown[e].pct}% (${stats.breakdown[e].detected}/${stats.breakdown[e].total})`),
                '',
                'DETALLE DE INSPECCIONES:',
                ...(turno.inspections.map(i =>
                  `  [${i.time}] ${i.result.total_persons} persona(s) · ${i.result.global_compliance}% cumplimiento` +
                  (i.result.persons.some(p => p.violations.length) ? ` · violaciones: ${Array.from(new Set(i.result.persons.flatMap(p => p.violations))).join(', ')}` : ' · OK')
                ))
              ]
              const b = new Blob([lines.join('\n')], { type: 'text/plain' })
              const a = document.createElement('a')
              a.href = URL.createObjectURL(b)
              a.download = `reporte_epp_${turno.date.replace(/\//g, '-')}.txt`
              a.click()
            }}>
            ↓ Descargar reporte del turno
          </button>
        </div>
      </main>
    </div>
  )
}
