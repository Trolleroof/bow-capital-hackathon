/**
 * minimap.ts — draws a top-down "COORDINATION MAP" of the swarm onto a 2D
 * canvas: the coverage grid (tinted covered cells) plus agent dots.
 */

export interface MiniAgent {
  x: number
  y: number
  alive: boolean
}

export interface MiniMarker {
  x: number
  y: number
  kind: 'target' | 'asset' | 'hostile' | 'rival'
  active?: boolean
}

export function drawMinimap(
  ctx: CanvasRenderingContext2D,
  size: number,
  worldHalf: number,
  grid: number,
  covered: Uint8Array,
  agents: MiniAgent[],
  markers: MiniMarker[] = [],
) {
  ctx.clearRect(0, 0, size, size)

  // background
  ctx.fillStyle = '#07120e'
  ctx.fillRect(0, 0, size, size)

  const cellPx = size / grid
  // covered cells. grid index = cx*grid + cy; cx -> screen x, cy -> screen y.
  ctx.fillStyle = 'rgba(78, 240, 160, 0.20)'
  for (let cx = 0; cx < grid; cx++) {
    for (let cy = 0; cy < grid; cy++) {
      if (covered[cx * grid + cy]) {
        ctx.fillRect(cx * cellPx, (grid - 1 - cy) * cellPx, cellPx, cellPx)
      }
    }
  }

  // grid lines
  ctx.strokeStyle = 'rgba(47, 174, 122, 0.18)'
  ctx.lineWidth = 1
  for (let i = 0; i <= grid; i++) {
    const p = i * cellPx
    ctx.beginPath()
    ctx.moveTo(p, 0)
    ctx.lineTo(p, size)
    ctx.moveTo(0, p)
    ctx.lineTo(size, p)
    ctx.stroke()
  }

  // agents — world (x,y) in [-worldHalf, worldHalf] -> canvas. y flipped so up
  // on screen = +y in world, matching the coverage tinting above.
  const markerInset = 4
  const toPx = (w: number) => {
    const px = ((w + worldHalf) / (2 * worldHalf)) * size
    return Math.max(markerInset, Math.min(size - markerInset, px))
  }

  for (const marker of markers) {
    if (marker.active === false) continue
    const px = toPx(marker.x)
    const py = size - toPx(marker.y)

    if (marker.kind === 'target') {
      ctx.strokeStyle = '#ffcf66'
      ctx.fillStyle = '#ffcf66'
      ctx.lineWidth = 1.5
      ctx.shadowColor = '#ffcf66'
      ctx.shadowBlur = 8
      ctx.beginPath()
      ctx.arc(px, py, 5.2, 0, Math.PI * 2)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(px - 7, py)
      ctx.lineTo(px + 7, py)
      ctx.moveTo(px, py - 7)
      ctx.lineTo(px, py + 7)
      ctx.stroke()
      ctx.shadowBlur = 0
    } else if (marker.kind === 'asset') {
      ctx.fillStyle = '#71d7ff'
      ctx.fillRect(px - 4, py - 4, 8, 8)
    } else {
      ctx.fillStyle = marker.kind === 'hostile' ? '#ff6b6b' : '#b48cff'
      ctx.beginPath()
      ctx.arc(px, py, 3.5, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  for (const a of agents) {
    const px = toPx(a.x)
    const py = size - toPx(a.y)
    ctx.beginPath()
    ctx.arc(px, py, a.alive ? 3.2 : 2.4, 0, Math.PI * 2)
    if (a.alive) {
      ctx.fillStyle = '#4ef0a0'
      ctx.shadowColor = '#4ef0a0'
      ctx.shadowBlur = 6
    } else {
      ctx.fillStyle = '#a44'
      ctx.shadowBlur = 0
    }
    ctx.fill()
    ctx.shadowBlur = 0
  }

  // frame
  ctx.strokeStyle = 'rgba(47, 174, 122, 0.6)'
  ctx.lineWidth = 1
  ctx.strokeRect(0.5, 0.5, size - 1, size - 1)
}
