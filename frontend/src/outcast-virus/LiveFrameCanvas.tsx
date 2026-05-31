import { useEffect, useRef } from 'react'
import type { SlamFrame } from './useOutcastVirusState'

interface Props {
  frame: SlamFrame | null
  className?: string
  fit?: 'cover' | 'contain'
  overlay?: (ctx: CanvasRenderingContext2D, width: number, height: number) => void
}

function drawFrame(
  canvas: HTMLCanvasElement,
  bitmap: ImageBitmap,
  fit: 'cover' | 'contain',
  overlay?: (ctx: CanvasRenderingContext2D, width: number, height: number) => void,
) {
  const parent = canvas.parentElement
  if (!parent) return

  const dpr = window.devicePixelRatio || 1
  const width = Math.max(1, Math.round(parent.clientWidth * dpr))
  const height = Math.max(1, Math.round(parent.clientHeight * dpr))

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
  }

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.clearRect(0, 0, width, height)
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'

  const scale = fit === 'cover'
    ? Math.max(width / bitmap.width, height / bitmap.height)
    : Math.min(width / bitmap.width, height / bitmap.height)
  const drawWidth = bitmap.width * scale
  const drawHeight = bitmap.height * scale
  const dx = (width - drawWidth) / 2
  const dy = (height - drawHeight) / 2

  ctx.drawImage(bitmap, dx, dy, drawWidth, drawHeight)
  overlay?.(ctx, width, height)
}

export function LiveFrameCanvas({ frame, className, fit = 'cover', overlay }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const bitmapRef = useRef<ImageBitmap | null>(null)
  const overlayRef = useRef<Props['overlay']>(overlay)
  const rafRef = useRef<number | null>(null)

  overlayRef.current = overlay

  const scheduleDrawRef = useRef<() => void>(() => {})
  scheduleDrawRef.current = () => {
    if (rafRef.current !== null) return
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null
      const canvas = canvasRef.current
      const bitmap = bitmapRef.current
      if (!canvas) return
      if (!bitmap) {
        const ctx = canvas.getContext('2d')
        ctx?.clearRect(0, 0, canvas.width, canvas.height)
        return
      }
      drawFrame(canvas, bitmap, fit, overlayRef.current)
    })
  }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    let cancelled = false

    if (!frame) {
      const ctx = canvas.getContext('2d')
      ctx?.clearRect(0, 0, canvas.width, canvas.height)
      bitmapRef.current?.close()
      bitmapRef.current = null
      scheduleDrawRef.current()
      return
    }

    ;(async () => {
      try {
        const bitmap = await createImageBitmap(frame.data)
        if (cancelled) {
          bitmap.close()
          return
        }
        bitmapRef.current?.close()
        bitmapRef.current = bitmap
        scheduleDrawRef.current()
      } catch {
        // Ignore transient decode failures on dropped/replaced frames.
      }
    })()

    return () => {
      cancelled = true
    }
  }, [frame, fit])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const redraw = () => {
      scheduleDrawRef.current()
    }

    const observer = new ResizeObserver(redraw)
    if (canvas.parentElement) {
      observer.observe(canvas.parentElement)
    }
    window.addEventListener('resize', redraw)

    return () => {
      observer.disconnect()
      window.removeEventListener('resize', redraw)
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
      bitmapRef.current?.close()
      bitmapRef.current = null
    }
  }, [fit])

  useEffect(() => {
    scheduleDrawRef.current()
  }, [overlay, fit, frame])

  return <canvas ref={canvasRef} className={className} />
}
