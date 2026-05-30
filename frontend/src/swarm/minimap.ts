/**
 * minimap.ts — draws a top-down "COORDINATION MAP" of the swarm onto a 2D
 * canvas: the coverage grid (tinted covered cells) plus agent dots.
 */

export interface MiniAgent {
  x: number
  y: number
  alive: boolean
}

export function drawMinimap(
  ctx: CanvasRenderingContext2D,
  size: number,
  worldHalf: number,
  grid: number,
  covered: Uint8Array,
  agents: MiniAgent[],
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
  const toPx = (w: number) => ((w + worldHalf) / (2 * worldHalf)) * size
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
