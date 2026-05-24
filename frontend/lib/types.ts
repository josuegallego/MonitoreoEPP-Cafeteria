export const EPP_CLASSES = ['delantal', 'gorro', 'guantes', 'tapabocas'] as const
export type EppClass = typeof EPP_CLASSES[number]

export const EPP_META: Record<EppClass, { color: string; icon: string; label: string }> = {
  delantal:  { color: '#4dabf7', icon: '', label: 'Delantal' },
  gorro:     { color: '#00e5b0', icon: '', label: 'Gorro' },
  guantes:   { color: '#ffb347', icon: '', label: 'Guantes' },
  tapabocas: { color: '#c084fc', icon: '', label: 'Tapabocas' },
}

export interface Person {
  id: number
  bbox: [number, number, number, number]
  conf: number
  detected: EppClass[]
  violations: EppClass[]
  probs: Record<EppClass, number>
  compliance: number
}

export interface DetectionResult {
  persons: Person[]
  global_compliance: number
  total_persons: number
  annotated_image: string   // base64 JPEG
}

export interface Inspection {
  id: string
  time: string
  timestamp: number
  result: DetectionResult
}
