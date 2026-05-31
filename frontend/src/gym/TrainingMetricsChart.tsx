import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import type { TrainingMetrics } from './TrainingDashboard'

interface TrainingMetricsChartProps {
  history: TrainingMetrics[]
}

export function TrainingMetricsChart({ history }: TrainingMetricsChartProps) {
  if (history.length === 0) {
    return (
      <div className="gym-metrics-chart-placeholder">
        <p>Waiting for training metrics...</p>
      </div>
    )
  }

  return (
    <div className="gym-metrics-chart">
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={history} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis
            dataKey="step"
            stroke="rgba(255,255,255,0.4)"
            tick={{ fontSize: 12 }}
            tickFormatter={(v) => (v % 200 === 0 ? String(v) : '')}
          />
          <YAxis stroke="rgba(255,255,255,0.4)" tick={{ fontSize: 12 }} width={40} />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(8, 15, 23, 0.9)',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: 4,
            }}
            labelStyle={{ color: 'rgba(255,255,255,0.8)' }}
          />
          <Legend wrapperStyle={{ paddingTop: 10 }} />
          <Line
            type="monotone"
            dataKey="reward"
            stroke="#4a9eff"
            dot={false}
            isAnimationActive={false}
            name="Reward"
          />
          <Line
            type="monotone"
            dataKey="task_score"
            stroke="#d5b76a"
            dot={false}
            isAnimationActive={false}
            name="Task Score"
          />
          <Line
            type="monotone"
            dataKey="coverage"
            stroke="#20c997"
            dot={false}
            isAnimationActive={false}
            name="Coverage"
          />
        </LineChart>
      </ResponsiveContainer>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={history} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis
            dataKey="step"
            stroke="rgba(255,255,255,0.4)"
            tick={{ fontSize: 12 }}
            tickFormatter={(v) => (v % 200 === 0 ? String(v) : '')}
          />
          <YAxis stroke="rgba(255,255,255,0.4)" tick={{ fontSize: 12 }} width={40} />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(8, 15, 23, 0.9)',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: 4,
            }}
            labelStyle={{ color: 'rgba(255,255,255,0.8)' }}
          />
          <Legend wrapperStyle={{ paddingTop: 10 }} />
          <Line
            type="monotone"
            dataKey="actor_loss"
            stroke="#ff6b6b"
            dot={false}
            isAnimationActive={false}
            name="Actor Loss"
          />
          <Line
            type="monotone"
            dataKey="critic_loss"
            stroke="#ffa500"
            dot={false}
            isAnimationActive={false}
            name="Critic Loss"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
